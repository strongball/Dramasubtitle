import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import VideoRNN, DecoderRNN, VideoEncoder, Context, EncoderRNN
from utils.tool import getLastOutputs, flatMutileLength

class ImgToSeq(nn.Module):
    def __init__(self, video_setting, decoder_setting, padding_idx=0):
        super(ImgToSeq, self).__init__()
        self.trainStep = 0
        self.videoRnn = VideoRNN(**video_setting)
        self.decoderRnn = DecoderRNN(**decoder_setting)
    def forward(self, input_img, input_seq, hidden=None):
        output, _ = self.makeContext(input_img)
        output, hidden = self.decoderRnn(input_seq, output[:,-1,:], hidden)
        return output, hidden
    
    def makeContext(self, input_img):
        output, hidden = self.videoRnn(input_img)
        return output, hidden
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.decoderRnn(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden
    
class OneImgToSeq(nn.Module):
    def __init__(self, video_setting, decoder_setting, padding_idx=0):
        super(OneImgToSeq, self).__init__()
        self.trainStep = 0
        self.videoRnn = VideoEncoder(**video_setting)
        self.decoderRnn = DecoderRNN(**decoder_setting)
    def forward(self, input_img, input_seq, hidden=None):
        output, _ = self.makeContext(input_img)
        output, hidden = self.decoderRnn(input_seq, output[:,-1,:], hidden)
        return output, hidden
    
    def makeContext(self, input_img):
        output, hidden = self.videoRnn(input_img)
        return output, hidden
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.decoderRnn(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden

class SubImgToSeq(nn.Module):
    def __init__(self, video_setting, subencoder_setting, decoder_setting, padding_idx=0):
        super(SubImgToSeq, self).__init__()
        self.trainStep = 0
        self.videoRnn = VideoEncoder(**video_setting)
        self.subRnn = EncoderRNN(**subencoder_setting)
        
        self.context = Context(video_feature = video_setting["output_size"], 
                               sub_feature =subencoder_setting["output_size"],
                               output_size = decoder_setting["feature_size"])
        
        self.decoderRnn = DecoderRNN(**decoder_setting)
        
    def forward(self, input_img, sub_seq, target_seq, hidden=None):
        output = self.makeContext(input_img, sub_seq)
        
        if(isinstance(target_seq, tuple)):
            target_seq, tarLengths = target_seq
        output, hidden = self.decoderRnn(target_seq, output, hidden)
        return output, hidden
    
    def makeContext(self, input_img, sub_seq):
        vout, _ = self.videoRnn(input_img)
        vout = vout[:,-1,:]
        
        if(isinstance(sub_seq, tuple)):
            subtitle, subLengths = sub_seq
        else:
            subtitle = sub_seq
            subLengths = [sub_seq.size(1)]*sub_seq.size(0)
            
        sout, _ = self.subRnn(subtitle)
        sout = getLastOutputs(sout, subLengths)
        
        cxt = self.context(vout, sout)
        return cxt
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.decoderRnn(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden
    
class SubToSeq(nn.Module):
    def __init__(self, subencoder_setting, decoder_setting, padding_idx=0):
        super(SubToSeq, self).__init__()
        self.trainStep = 0
        self.subRnn = EncoderRNN(**subencoder_setting)
        
        self.decoderRnn = DecoderRNN(**decoder_setting)
        
    def forward(self, sub_seq, target_seq, hidden=None):
        output = self.makeContext(sub_seq)
        
        if(isinstance(target_seq, tuple)):
            target_seq, tarLengths = target_seq
        output, hidden = self.decoderRnn(target_seq, output, hidden)
        return output, hidden
    
    def makeContext(self, sub_seq):
        if(isinstance(sub_seq, tuple)):
            subtitle, subLengths = sub_seq
        else:
            subtitle = sub_seq
            subLengths = [sub_seq.size(1)]*sub_seq.size(0)
            
        sout, _ = self.subRnn(subtitle)
        sout = getLastOutputs(sout, subLengths)

        return sout
    
    def decode(self, input_seq, context, hidden=None):
        output, hidden = self.decoderRnn(input_seq, context, hidden)
        output = F.softmax(output, dim=2)
        return output, hidden