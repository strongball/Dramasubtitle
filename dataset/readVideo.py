import cv2
from PIL import Image
import json
import os
import glob

class ReadSubtitle():
    def __init__(self, videoFile, subFile, order, transform=None, inter=1, maxFrame=128, timeOffset=0, useBmp=False):
        with open(subFile, 'r') as f:
            self.data = json.load(f)
        self.pairs = []
        for i in range(len(self.data)-1):
            if self.data[i+1]["start"] - self.data[i]["end"] < 1:
                self.pairs.append(i)
        self.transform = transform
        self.videoFile = videoFile
        self.inter = inter
        self.timeOffset = timeOffset
        self.maxFrame = maxFrame
        self.order = order
        self.useBmp = useBmp
        
    def __getitem__(self, index):
        ann = self.data[self.pairs[index]]
        imgs = []

        if self.maxFrame > 0:
            if self.useBmp: 
                imgs, sucess = self.getBmp(ann["start"])
            else:
                imgs, sucess = self.getFrames(ann["start"], ann["end"])

            if not sucess:
                print("Get frame error!!\nTime: {}, At: {}".format(ann["start"], self.videoFile))
                if index > 0:
                    return self[index-1]
                else:
                    return self[index+1]

        return self.data[self.pairs[index]]["sub"], self.data[self.pairs[index]+1]["sub"], imgs
    
    def __len__(self):
        return len(self.pairs)
    
    def getBmp(self, start):
        dirName = os.path.join(self.videoFile, self.order)
        file = os.path.join(dirName, str(start))+".bmp"
        return [Image.open(file)], True
        
    def getFrames(self, start, end):
        imgs = []
        fsize = 0
        retry = 0
        time = start + self.timeOffset * (end-start)
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
    def __init__(self, basedir, inter=1, maxSeries=None, maxFrame = 128, transform=None, timeOffset=0, useBmp=False):
        self.dataFiles = []
        videoDir = os.path.join(basedir, "video")
        subDir = os.path.join(basedir, "json")
        self.frameDir = os.path.join(basedir, "frames")
        self.inter = inter
        self.timeOffset = timeOffset
        self.maxFrame = maxFrame
        
        self.epochs = []
        sx = 1
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
                                                    maxFrame = self.maxFrame, 
                                                    timeOffset = self.timeOffset,
                                                    useBmp=useBmp,
                                                    transform=transform))
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