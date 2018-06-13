import cv2
from PIL import Image
import pandas as pd
import os
import glob
import random

class ReadSubtitle():
    def __init__(self, videoFile, subFile, order, pairTime=1, randomStart=False, transform=None, inter=1, maxFrame=128, timeOffset=0, useBmp=False, subOffset=0, subMax=None):
        self.data = self.makeDataFrame(subFile)
        self.data = self.data[self.data["inter_time"] < pairTime]
        self.data = self.data[subOffset : subMax]
        self.transform = transform
        self.videoFile = videoFile
        self.inter = inter
        self.timeOffset = timeOffset
        self.randomStart = randomStart
        self.maxFrame = maxFrame
        self.order = order
        self.useBmp = useBmp
        
        
    def makeDataFrame(self, fileName):
        df = pd.read_json(fileName)
        df = df.sort_values("start")
        df["start"] = df["start"].round(decimals=3)
        df["end"] = df["end"].round(decimals=3)
        df["inter_time"] = df["start"].shift(-1) - df["end"]
        df["nsub"] = df["sub"].shift(-1)
        
        df = df[df["inter_time"] >= 0]
        return df
    
    def __getitem__(self, index):
        ann = self.data.iloc[index]
        imgs = []

        if self.maxFrame > 0:
            if self.useBmp: 
                imgs, sucess = self.getBmp(ann["start"])
            else:
                imgs, sucess = self.getFrames(ann["start"], ann["end"])

            if not sucess:
                print("Get frame error!!\nTime: {}, Index: {}, At: {}".format(ann["start"], index, self.order))
                if index > 0:
                    return self[index-1]
                else:
                    return self[index+1]

        return ann["sub"], ann["nsub"], imgs
    
    def __len__(self):
        return len(self.data)
    
    def getBmp(self, start):
        dirName = os.path.join(self.videoFile, self.order)
        file = os.path.join(dirName, str(start))+".bmp"
        
        if os.path.isfile(file):
            img = Image.open(file)
            if self.transform is not None:
                img = self.transform(img)
            return [img], True
        else:
            return [], False
        
    def getFrames(self, start, end):
        imgs = []
        fsize = 0
        retry = 0
        
        time = start + self.timeOffset * (end-start)
        if self.randomStart:
            time = random.uniform(time, end)
        inter = (end - time) / self.maxFrame
        
        cap = cv2.VideoCapture(self.videoFile)
        while self.maxFrame > fsize and time < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, time*1000)
            sucess, img = cap.read()
            if sucess:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
                time += inter
                fsize += 1
            else:
                retry += 1
                if retry > 10:
                    print("Get frame error!!\nTime: {}, At: {}".format(start, self.videoFile))
                    return imgs, False
        return imgs, True
                    
class DramaDataset():
    def __init__(self, basedir, pairTime=1, inter=1, randomStart=False, startSeries=1, maxSeries=None, maxFrame = 128, transform=None, timeOffset=0, useBmp=False, subOffset=0, subMax=None):
        self.dataFiles = []
        videoDir = os.path.join(basedir, "video")
        subDir = os.path.join(basedir, "json")
        self.frameDir = os.path.join(basedir, "frames")
        self.inter = inter
        self.randomStart = randomStart
        self.timeOffset = timeOffset
        self.maxFrame = maxFrame
        
        self.epochs = []
        sx = startSeries
        while True:
            notEpoch = 0
            serExist = False
            ex = 1
            while True:
                ep = "S%02dE%02d"%(sx,ex)
                videoFiles = glob.glob(os.path.join(videoDir, "*%s*"%(ep)))
                subFiles = glob.glob(os.path.join(subDir, "*%s*"%(ep)))
                if len(videoFiles) > 0 and len(subFiles) > 0:
                    videoF = self.frameDir if useBmp else videoFiles[0]
                    self.epochs.append(ReadSubtitle(videoF, 
                                                    subFiles[0],
                                                    order = ep,
                                                    inter = self.inter,
                                                    pairTime=pairTime,
                                                    randomStart = self.randomStart,
                                                    maxFrame = self.maxFrame, 
                                                    timeOffset = self.timeOffset,
                                                    useBmp=useBmp,
                                                    transform=transform,
                                                    subOffset=subOffset, 
                                                    subMax=subMax))
                    notEpoch = 0
                    serExist = True
                else:
                    notEpoch += 1
                    if notEpoch >= 2:
                        break
                ex += 1 
            if not serExist or (maxSeries and maxSeries <= sx):
                break
            sx += 1
        print("Total Drama: {}".format(len(self.epochs)))
        self.indexTable = []
        itemSize = 0
        for subset in self.epochs:
            self.indexTable.append(itemSize)
            itemSize += len(subset)
        self.indexTable.append(itemSize) # add total size
        
    def __getitem__(self, index):
        subindex = -1
        for ep in range(len(self.indexTable)-1):
            if index >= self.indexTable[ep] and index < self.indexTable[ep+1]:
                subindex = index - self.indexTable[ep]
                break
        if subindex == -1:
            raise IndexError("Out of range!! Index: {}".format(index))
        return self.epochs[ep][subindex]
    
    def __len__(self):
        return self.indexTable[-1]