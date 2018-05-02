import torch 
import torch.nn as nn

from model.model import EncoderRNN
from utils.tool import getLastOutputs

class Gesd(nn.Module):
    def __init__(self, gamma=1, c=1):
        super(Gesd, self).__init__()
        self.gamma = gamma
        self.c = c

    def forward(self, f1,f2):
        l2_norm = ((f1-f2) ** 2).sum(dim=1)
        euclidean = 1 / (1 + l2_norm)
        sigmoid  = 1 / (1 + torch.exp(-1 * self.gamma * ((f1*f2).sum(dim=1) + self.c)))
        output = euclidean * sigmoid

        return output
    
class GesdSimilarity(nn.Module):
    def __init__(self, subencoder_setting, gamma=1, c=1):
        super(GesdSimilarity, self).__init__()
        self.preEncoder = EncoderRNN(**subencoder_setting)
        self.nextEncoder = EncoderRNN(**subencoder_setting)
        self.gesd = Gesd(gamma=gamma, c=c)

    def forward(self, s1, s2):
        if(isinstance(s1, tuple)):
            s1Sentence, s1Lengths = s1
        else:
            s1Sentence = s1
            s1Lengths = [s1.size(1)]*s1.size(0)
            
        if(isinstance(s2, tuple)):
            s2Sentence, s2Lengths = s2
        else:
            s2Sentence = s2
            s2Lengths = [s2.size(1)]*s2.size(0)
            
        feature1, _ = self.preEncoder(s1Sentence)
        feature2, _ = self.nextEncoder(s2Sentence)
        
        feature1 = getLastOutputs(feature1, s1Lengths)
        feature2 = getLastOutputs(feature2, s2Lengths)
        
        output = self.gesd(feature1, feature2)
        return output