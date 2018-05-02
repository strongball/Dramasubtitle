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
def trainer(args):
    modelDir = args.model
    LangFile = os.path.join(modelDir, "Lang.pkl")
    modelFile = args.checkpoint
    
    MaxEpoch = args.epoch
    BatchSize = args.batch
    DataDir = args.data
    lr = args.lr
    
    use_cuda = torch.cuda.is_available()
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
        global Variable
        Variable = Variable.cuda
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    recLoss = Average()
    timer = Timer()
    trainStep = 0
    for epoch in range(MaxEpoch):
        for i, data in enumerate(loader, 1):
            try:
                pre, nex, imgs = data
                pre, nex, scores = makeNegSample(pres, nexs, negSize=2)
                loss = step(model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            subtitles= pre,
                            targets=nex,
                            scores=scores,
                            lang=lang)

                recLoss.addData(loss.data[0])
                writer.add_scalar('loss', loss.data[0], trainStep)
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
    imgs, inSubtitles, inTargets, outTargets = transInputs(imgs, subtitles, targets, lang)
    inSubtitles = sentenceToVector(subtitles, lang)
    inTargets = sentenceToVector(targets, lang)
    scores = Variable(torch.Tensor(scores))
    
    ouputs = model(inSubtitles, inTargets)
    
    loss = criterion(ouputs, scores)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def sentenceToVector(sentences, lang, sos=False, eos=False):
    vectors = []
    vectorTransforms = [torch.LongTensor, Variable]
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
            "output_size": 1024 
        }
        model = GesdSimilarity(subencoderOpt)
        
    return lang, model

def predit(model, lang, imgs, subtitle,max_length=50):
    ans = []
    inputImgs = Variable(imgs.unsqueeze(0))
    subtitle = Variable(torch.LongTensor(lang.sentenceToVector(subtitle, sos=False, eos=False)).unsqueeze(0))
    inputs = Variable(torch.LongTensor([[lang["SOS"]]]).long())
    hidden = None
    
    cxt = model.makeContext(inputImgs, subtitle)
    for i in range(max_length):
        outputs, hidden = model.decode(inputs, cxt, hidden)
        prob, outputs = outputs.topk(1)
        outputs = outputs[0][0].data[0]
        if(outputs == lang["EOS"]):
            break
        ans.append(outputs)
        inputs = Variable(torch.LongTensor([[outputs]]))
    return lang.vectorToSentence(ans)

if __name__ == "__main__":
    args = parser.parse_args()
    trainer(args)
