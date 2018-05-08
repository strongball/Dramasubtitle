import torch
import torch.nn as nn

from torch import optim
import torchvision.transforms as transforms
import torch.utils.data

from utils.CvTransform import CvResize, CvCenterCrop
from utils.tokenMaker import Lang
from utils.tool import padding, flatMutileLength, Timer, Average
from model.BigModel import SubImgToSeq, SubVideoToSeq
from dataset.readVideo import DramaDataset

from tensorboardX import SummaryWriter

import os
import time
import pickle
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
parser.add_argument('-b', '--batch', help="Batch size", type=int, default=30)
parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
parser.add_argument('-m', '--model', help="model dir", required=True)
parser.add_argument('-d', '--data', help="Data loaction", required=True)

splt = " "
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainer(args):
    modelDir = args.model
    LangFile = os.path.join(modelDir, "Lang.pkl")
    modelFile = args.checkpoint
    
    MaxEpoch = args.epoch
    BatchSize = args.batch
    DataDir = args.data
    lr = args.lr
    
    print("=========Use Device: {}=========\n".format(device))
    lang, model = Loadmodel(modelDir,
                            LangFile,
                            modelFile,
                            dataDir=DataDir)
    
    datasets = DramaDataset(basedir=DataDir,
                            maxFrame=1,
                            timeOffset=0.2,
                            transform = transforms.Compose([CvResize(256),
                                                            CvCenterCrop(224),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                                 [0.229, 0.224, 0.225])]))
    loader = torch.utils.data.DataLoader(datasets, batch_size=BatchSize, shuffle=True, num_workers=4)
    print("Data size\t: {}".format(len(datasets)))
    print("Max epoch\t: {}\nBatch size\t: {}\nLearning rate\t: {}\n".format(MaxEpoch, BatchSize, lr))
    print("Start training........\n")
    writer = SummaryWriter(modelDir)
    
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    recLoss = Average()
    timer = Timer()
    trainStep = 0
    for epoch in range(MaxEpoch):
        for i, data in enumerate(loader, 1):
            try:
                pre, nex, imgs = data
                loss = step(model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              imgs=imgs[0],
                             subtitles= pre,
                              targets=nex,
                             lang=lang)

                recLoss.addData(loss.item())
                writer.add_scalar('loss', loss.item(), trainStep)
                trainStep += 1
                loss = None
                if i % 50 == 0:                        
                    print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                          .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
                    pred = predit(model, lang, imgs[0][:1], pre[0])
                    print("F: {}\nS: {}\nP: {}\n"
                          .format(pre[0], nex[0], pred))
            except Exception as exp:
                print("Step error: {}".format(i))
                print(exp)
        if i % 50 != 0:
            print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                  .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
        modelName = os.path.join(modelDir, "SubImgModel.{}.pth".format(int((epoch+1)/5)))
        print("Saving Epoch model: {}.....\n".format(modelName))
        torch.save(model, modelName)

def step(model, criterion, optimizer, imgs, subtitles, targets, lang):
    imgs, inSubtitles, inTargets, outTargets = transInputs(imgs, subtitles, targets, lang)
    
    outputs, hidden = model(imgs, inSubtitles, inTargets)
    
    outputs = flatMutileLength(outputs, outTargets[1])
    targets = flatMutileLength(outTargets[0], outTargets[1])
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def transInputs(imgs, subtitles, targets, lang):
    if imgs.dim() < 5:
        imgs = imgs.unsqueeze(1)
    imgs = imgs.to(device)
        
    inSubtitles = []
    inTargets = []
    outTargets = []
    
    vectorTransforms = [lambda x: torch.LongTensor(x).to(device)]
    
    for subtitle in subtitles:
        inSubtitles.append(lang.sentenceToVector(subtitle, sos=False, eos=False))
    inSubtitles = padding(inSubtitles, lang["PAD"], vectorTransforms)
    
    for target in targets:
        inTargets.append(lang.sentenceToVector(target, sos=True, eos=False))
        outTargets.append(lang.sentenceToVector(target, sos=False, eos=True))
    inTargets = padding(inTargets, lang["PAD"], vectorTransforms)
    outTargets = padding(outTargets, lang["PAD"], vectorTransforms)

    return imgs, inSubtitles, inTargets, outTargets

def createLang(name, dataDir):
    lang = Lang(name, splt)
    smalldatasets = DramaDataset(basedir=dataDir, maxFrame=0)
    
    print("Data size: {}".format(len(smalldatasets)))
    timer = Timer()
    for s1, s2, _ in smalldatasets:
        lang.addSentance(s1)
        lang.addSentance(s2)

    print("Create lang model. Number of word: {}".format(len(lang)))
    print("Total time: {:.2f}".format(timer.getTime()))
    return lang

def Loadmodel(modelDir, LangBag, modelfile, dataDir):
    if not os.path.isdir(modelDir):
        os.mkdir(modelDir)

    if os.path.isfile(LangBag):
        with open(LangBag, 'rb') as f:
            lang = pickle.load(f)
            print("Load lang model: {}. Word size: {}".format(LangBag, len(lang)))
    else:
        lang = createLang(LangBag, dataDir)
        with open(LangBag, 'wb') as f:
            pickle.dump(lang, f)
          
    if os.path.isfile(modelfile):
        model = torch.load(modelfile)
        print("Load model: {}.".format(modelfile))
    else:
        videoOpt = {
            "cnn_hidden": 1024,
            "hidden_size": 512,
            "output_size": 1024,
            #"num_layers": 1,
            "dropout": 0.1,
            "pretrained": True
        }
        subencoderOpt = {
            "word_size": len(lang),
            "em_size": 256,
            "num_layers": 2,
            "dropout": 0.1,
            "hidden_size": 256,
            "output_size": 512 
        }
        decoderOpt = {
            "word_size": len(lang),
            "em_size": 256,
            "num_layers": 2,
            "dropout": 0.1,
            "hidden_size": 256,
            "feature_size": 1024 
        }
        model = SubImgToSeq(videoOpt, subencoderOpt, decoderOpt)
        #model = SubVideoToSeq(videoOpt, subencoderOpt, decoderOpt)
        
    return lang, model

def predit(model, lang, imgs, subtitle,max_length=50):
    ans = []
    inputImgs = imgs.unsqueeze(0).to(device)
    subtitle = torch.LongTensor(lang.sentenceToVector(subtitle, sos=False, eos=False)).unsqueeze(0).to(device)
    inputs = torch.LongTensor([[lang["SOS"]]]).to(device)
    
    hidden = None
    
    cxt = model.makeContext(inputImgs, subtitle)
    for i in range(max_length):
        outputs, hidden = model.decode(inputs, cxt, hidden)
        prob, outputs = outputs.topk(1)
        if(outputs.item() == lang["EOS"]):
            break
        ans.append(outputs.item())
        
        inputs = outputs.squeeze(1).detach()
    return lang.vectorToSentence(ans)

if __name__ == "__main__":
    args = parser.parse_args()
    trainer(args)
