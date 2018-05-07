import torch
import torch.nn as nn

from torch import optim
from torch.autograd import Variable
import torch.utils.data

from utils.tokenMaker import Lang
from utils.tool import padding, flatMutileLength, Timer, Average
from Similarity.model import GesdSimilarity
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
use_cuda = torch.cuda.is_available()
def trainer(args):
    modelDir = args.model
    LangFile = os.path.join(modelDir, "Lang.pkl")
    modelFile = args.checkpoint
    
    MaxEpoch = args.epoch
    BatchSize = args.batch
    DataDir = args.data
    lr = args.lr
    
    print("=========Use GPU: {}=========\n".format(use_cuda))
    lang, model = Loadmodel(modelDir,
                            LangFile,
                            modelFile,
                            dataDir=DataDir)
    
    datasets = DramaDataset(basedir=DataDir,
                            maxFrame=0,
                            timeOffset=0.2,)
    loader = torch.utils.data.DataLoader(datasets, batch_size=BatchSize, shuffle=True, num_workers=4)
    print("Data size\t: {}".format(len(datasets)))
    print("Max epoch\t: {}\nBatch size\t: {}\nLearning rate\t: {}\n".format(MaxEpoch, BatchSize, lr))
    print("Start training........\n")
    writer = SummaryWriter(modelDir)
    
    if use_cuda:
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    recLoss = Average()
    timer = Timer()
    trainStep = 0
    for epoch in range(MaxEpoch):
        for i, data in enumerate(loader, 1):
            #try:
                pre, nex, imgs = data
                pre, nex, scores = makeNegSample(pre, nex, negSize=2)
                loss = step(model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            subtitles= pre,
                            targets=nex,
                            scores=scores,
                            lang=lang)

                recLoss.addData(loss.item())
                writer.add_scalar('loss', loss.item(), trainStep)
                trainStep += 1
                loss = None
                if i % 50 == 0:                        
                    print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                          .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
            #except Exception as exp:
                #print("Step error: {}".format(i))
                #print(exp)
        if i % 50 != 0:
            print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                  .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
        modelName = os.path.join(modelDir, "SimilarityModel.{}.pth".format(int((epoch+1)/5)))
        print("Saving Epoch model: {}.....\n".format(modelName))
        torch.save(model, modelName)

def makeNegSample(pres, nexs, negSize):
    mpres = []
    mnexts = []
    scores = []
    for pre, nex in zip(pres, nexs):
        while True:
            negs = random.sample(nexs, negSize)
            if not nex in negs:
                break
        mpres +=[pre] * (negSize + 1)
        mnexts += [nex] + negs
        scores += [1] + [0] * negSize
    return mpres, mnexts, scores

def step(model, optimizer, criterion, subtitles, targets, scores, lang):
    inSubtitles = sentenceToVector(subtitles, lang)
    inTargets = sentenceToVector(targets, lang)
    scores = torch.cuda.FloatTensor(scores) if use_cuda else torch.FloatTensor(scores)
    scores = Variable(scores)
    
    ouputs = model(inSubtitles, inTargets)
    
    loss = criterion(ouputs, scores)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def sentenceToVector(sentences, lang, sos=False, eos=False):
    vectors = []
    vectorTransforms = [torch.cuda.LongTensor, Variable] if use_cuda else [torch.LongTensor, Variable]
    for s in sentences:
        vectors.append(lang.sentenceToVector(s, sos=sos, eos=eos))
    vectors = padding(vectors, lang["PAD"], vectorTransforms)
    return vectors

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
        subencoderOpt = {
            "word_size": len(lang),
            "em_size": 512,
            "hidden_size": 512,
            "output_size": 512 
        }
        model = GesdSimilarity(subencoderOpt)
        
    return lang, model

if __name__ == "__main__":
    args = parser.parse_args()
    trainer(args)
