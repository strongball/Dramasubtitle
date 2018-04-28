import torch
import time
import re

punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
repunctuation = re.compile('[%s]' % re.escape(punctuation))
def fixString(s):
    return re.sub(repunctuation, "", s).lower()

def padding(seqs, pad, transforms=None, sort=False):
    if sort:
        seqs.sort(key=len, reverse=True)
        
    lengths = [len(s) for s in seqs]
    maxLength = max(lengths)
    for s, l in zip(seqs, lengths):
        s += [pad]*(maxLength-l)
    if transforms:
        if isinstance(transforms, list):
            for transform in transforms:
                seqs = transform(seqs)
        else:
            seqs = transforms(seqs)
    return seqs, lengths

def getLastOutputs(datas, lengths):
    lasts = []
    for data, length in zip(datas, lengths):
        lasts.append(data[length-1])
    return torch.stack(lasts)

def flatMutileLength(datas, lengths):
    flat = []
    for data, length in zip(datas, lengths):
        flat.append(data[:length])
    flat = torch.cat(flat)
    return flat

class Timer():
    def __init__(self):
        self.startTime = time.time()
        
    def getTime(self):
        return time.time() - self.startTime
    
    def reset(self):
        self.startTime = time.time()
        
    def getAndReset(self):
        now = time.time()
        inter = now - self.startTime
        self.startTime = now
        return inter
    
class Average():
    def __init__(self):
        self.datas = []
        
    def addData(self, data):
        self.datas.append(data)
        
    def mean(self):
        mean = sum(self.datas) / len(self.datas)
        return mean
    
    def reset(self):
        self.datas = []
        
    def getAndReset(self):
        mean = self.mean()
        self.reset()
        return mean