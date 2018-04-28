import re
import chardet
import json
import os
import glob
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-s", "--srt", help="Srt Dir file location", required=True)
parser.add_argument("-o", "--out", help="Output json location", required=True)

stopSentence = ["校对", "www", "翻译组", "翻译：","字幕", "压制", "压缩", "上集", "■", "时间轴", "font color"]
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
toSpace = re.compile('[%s\s]+' % re.escape(punctuation))

ftr = [3600,60,1,0.001]

def srtTime2second(srtTime):
    return sum([a*b for a,b in zip(ftr, map(int, re.split("[:,]", srtTime)))])

def cleanString(s):
    s = s.replace("\n", "")
    s = re.sub(toSpace, " ", s)
    s = re.sub("^\s*|\s*$", "", s)
    return s

def srt2json(srtfile, jsonfile):
    try:
        with open(srtfile, 'rb') as f:
            result = chardet.detect(f.read())
        with open(srtfile, 'r', encoding=result['encoding']) as f:
            parts = []
            tmp = []
            for l in f.readlines():
                if l == '\n':
                    if len(tmp) > 2:
                        parts.append(tmp)
                        tmp = []
                else:
                    tmp.append(l)
    except Exception as e:
        return None               
    dictType = []
    for old in parts:
        r = {}
        clean = True
        times = re.split("\n|-->", old[1])
        r["start"] =  srtTime2second(times[0])
        r["end"] =  srtTime2second(times[1])
        r["sub"] = "".join(old[2])
        r["sub"] = cleanString(r["sub"])
        for stop in stopSentence:
            if stop in r["sub"]:
                clean = False
                break
        if clean:
            dictType.append(r)
    with open(jsonfile, "w", encoding='utf8') as outfile:
        json.dump(dictType, outfile,ensure_ascii=False)
    return len(dictType)
        
if __name__ == '__main__':
    args = parser.parse_args()
    srtDir = args.srt
    jsonDir = args.out
    
    if not os.path.isdir(jsonDir):
        os.makedirs(jsonDir)
    
    for filename in glob.glob(os.path.join(srtDir, '*.srt')):
        outputname = os.path.splitext(os.path.basename(filename))[0] + ".json"
        print("Make: ",filename)
        size = srt2json(filename, os.path.join(jsonDir, outputname))
        if size:
            print("Make file: {:15}, size: {:4}".format(outputname, size))
        else:
            print("Error: {}".format(filename))
    
    
    
    
    
    