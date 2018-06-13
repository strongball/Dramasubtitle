import re
import chardet
import json
import os
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="Srt Dir file location", required=True)

punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~？！－'
toSpace = re.compile('[{}\s]+'.format(re.escape(punctuation)))
def cleanString(s):
    s = re.sub("<[^<>]*>", "SPECTAG", s) # remove xml tags
    s = re.sub("{[^<>]*}", "SPECTAG", s) # remove {} tags
    s = re.sub(toSpace, " ", s)
    s = re.sub("^\s+|\s+$", "", s)
    return s

ftr = [3600,60,1,0.001]
def srtTime2second(srtTime):
    return sum([a*b for a,b in zip(ftr, map(int, re.split("[:,]", srtTime)))])

stopSentence = "|".join(["以下内容为电视台提示", "人人影視", "SPECTAG", "目 前 資 金", "最 新 余 額", "影音工作室", "影视论坛", "校对", "后期", "翻校", "時間軸", "www", "翻译组", "翻译","字幕", "压制", "压缩", "上集", "■", "时间轴", "font color", "第.季", "第.集", "AVI"])
endToken = re.compile('^\n')

def srt2json(srtfile, jsonfile):
    with open(srtfile, 'rb') as f:
        result = chardet.detect(f.read())
    with open(srtfile, 'r', encoding=result['encoding']) as f:
        data = f.readlines()
    
    startTime = 0
    endTime = 0
    parts = []
    row = None
    for s in data:
        # ignore srt number or space
        if re.match("\d+\n", s):
            continue

        # check time
        rtime = re.findall("\d{2}:\d{2}:\d{2},\d{3}", s)
        if len(rtime) == 2:
            startTime = srtTime2second(rtime[0])
            endTime = srtTime2second(rtime[1])
            continue

        # end of sentence
        if re.search(endToken, s):
            if row:
                row['end'] = endTime
                if not re.search(stopSentence, row['sub']) and len(row['sub']) > 0:
                    parts.append(row)
            row = None
        else:
            s = cleanString(s)
            if row is None:
                row = {}
                row['start'] = startTime
                row['sub'] = s
            else:
                row['sub'] += " " + s
    with open(jsonfile, "w", encoding='utf8') as outfile:
        json.dump(parts, outfile, ensure_ascii=False)
    return len(parts)

if __name__ == '__main__':
    args = parser.parse_args()
    srtDir = os.path.join(args.data, "sub")
    jsonDir = os.path.join(args.data, "json")
    
    if not os.path.isdir(jsonDir):
        os.makedirs(jsonDir)
    print(len(glob.glob(os.path.join(srtDir, '*.srt'))))
    for filename in glob.glob(os.path.join(srtDir, '*.srt')):
        outputname = os.path.splitext(os.path.basename(filename))[0] + ".json"
        print("Start file: {:15}".format(filename))
        size = srt2json(filename, os.path.join(jsonDir, outputname))
        if size > 100:
            print("Make file: {:15}, size: {:4}".format(outputname, size))
        else:
            print("Error\t: {}\tSize: {}".format(filename, size))
    
    
    
    
    
    