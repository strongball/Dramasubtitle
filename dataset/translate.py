import pickle

class TranslateDataset():
    def __init__(self, basedir, pairTime=1, inter=1, randomStart=False, startSeries=1, maxSeries=None, maxFrame = 128, transform=None, timeOffset=0, useBmp=False, subOffset=0, subMax=None):
        with open(basedir, 'rb') as f:
            self.trainset = pickle.load(f)
        self.trainset = self.trainset[startSeries : maxSeries]
    def __getitem__(self, index):
        f,e = self.trainset[index]
        return f, e, []
    
    def __len__(self):
        return len(self.trainset)