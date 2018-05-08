import re
import chardet
import json
import os
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="Srt Dir file location", required=True)

punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
toSpace = re.compile('[{}\s]+'.format(re.escape(punctuation)))
def cleanString(s):
    s = re.sub("<[^<>]*>", "", s) # remove xml tags
    s = re.sub(toSpace, " ", s)
    s = re.sub("^\s+|\s+$", "", s)
    return s

ftr = [3600,60,1,0.001]
def srtTime2second(srtTime):
    return sum([a*b for a,b in zip(ftr, map(int, re.split("[:,]", srtTime)))])

stopSentence = "|".join(["www", "font color", "</font>", "♪"])
endToken = re.compile('[{}]'.format(re.escape('!?.')))
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
        if re.match("\d*\n", s) or re.search(stopSentence, s):
            continue

        # check time
        rtime = re.findall("\d{2}:\d{2}:\d{2},\d{3}", s)
        if len(rtime) == 2:
            startTime = srtTime2second(rtime[0])
            endTime = srtTime2second(rtime[1])
            continue

        # end of sentence
        if re.search(endToken, s):
            s = cleanString(s)
            if row is None:
                row = {}
                row['start'] = startTime
                row['sub'] = ""
            row['sub'] += s
            row['end'] = endTime
            parts.append(row)
            row = None
        else:
            s = cleanString(s)
            if row is None:
                row = {}
                row['start'] = startTime
                row['sub'] = ""
            row['sub'] += s + " "
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
        size = srt2json(filename, os.path.join(jsonDir, outputname))
        if size > 300:
            print("Make file\t: {:15}, size: {:4}".format(outputname, size))
        else:
            print("Error\t: {}".format(filename))