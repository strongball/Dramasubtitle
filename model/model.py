import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.models as models

import torchvision.transforms as transforms

    
class EncoderRNN(nn.Module):
    def __init__(self, word_size, em_size, hidden_size, output_size, num_layers=1, dropout=0.5, rnn_type="GRU", padding_idx=0):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(word_size, em_size, padding_idx=padding_idx)
        
        if rnn_type == "GRU":
            rnn_type = nn.GRU
        elif rnn_type == "LSTM":
            rnn_type = nn.LSTM
        self.gru = rnn_type(em_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        if input.dim() != 2:
            raise Exception("Encoder dim error. (batch, step)")
        output = self.embedding(input)
        self.gru.flatten_parameters()
        #output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = F.relu(output)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, word_size, em_size, hidden_size, feature_size, num_layers=1, dropout=0.5, padding_idx=0):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(word_size, em_size, padding_idx=0)
        self.gru = nn.GRU(em_size + feature_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, word_size)

    def forward(self, input, feature, hidden=None):
        if input.dim() != 2:
            raise Exception("DecoderRNN dim error. (batch, step)")
            
        step = input.size(1)
        
        output = self.embedding(input)
        #output = F.relu(output)
        feature = torch.stack([feature]*step, 1)#copy featurn to (batch, step, feature)
        output = torch.cat((output, feature), 2)
        self.gru.flatten_parameters()
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = F.relu(output)
        return output, hidden

class VideoRNN(nn.Module):
    def __init__(self, cnn_hidden, hidden_size, num_layers=1, output_size=512, dropout=0.5, cnn_type=models.resnet50, rnn_type="GRU", pretrained=True):
        super(VideoRNN, self).__init__()
        self.cnn = cnn_type(pretrained=pretrained)
        if cnn_type == models.resnet50:
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, cnn_hidden)
        if rnn_type == "GRU":
            rnn_type = nn.GRU
        elif rnn_type == "LSTM":
            rnn_type = nn.LSTM
        self.rnn = rnn_type(cnn_hidden, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, imgs, hidden=None):
        batch = imgs.size(0)
        step = imgs.size(1)
        imgs = imgs.view(batch*step, 3, 224, 224)#(batch, images)
        output = self.cnn(imgs)
        output = F.dropout(output)
        output = F.relu(output)
        
        output = output.view(batch, step, -1)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        output = F.relu(output)
        return output, hidden
    
class Context(nn.Module):
    def __init__(self, video_feature, sub_feature, output_size):
        super(Context, self).__init__()
        self.fc = nn.Linear(video_feature+sub_feature, output_size)

    def forward(self, videoFeature, subFeature):
        output = torch.cat((videoFeature, subFeature), 1)
        output = self.fc(output)
        output = F.relu(output)
        return output
    
class VideoEncoder(nn.Module):
    def __init__(self, cnn_hidden, hidden_size, output_size=512, cnn_type=models.resnet50, pretrained=True):
        super(VideoEncoder, self).__init__()
        self.cnn = cnn_type(pretrained=pretrained)
        if cnn_type == models.resnet50:
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, cnn_hidden)
        self.out = nn.Linear(cnn_hidden, output_size)

    def forward(self, imgs, hidden=None):
        batch = imgs.size(0)
        step = imgs.size(1)
        imgs = imgs.view(batch*step, 3, 224, 224)#(batch, images)
        output = self.cnn(imgs)
        output = F.dropout(output)
        output = F.relu(output)
        
        output = self.out(output)
        output = F.dropout(output)
        output = F.relu(output).unsqueeze(1)
        return output, hidden