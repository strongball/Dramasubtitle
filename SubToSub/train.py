import torch
import torch.nn as nn
from torch import optim
import torch.utils.data

from model.BigModel import SubToSeqFix as SubToSeq
from utils.tokenMaker import Lang
from utils.tool import padding, flatMutileLength, Timer, Average
from dataset.readVideo import DramaDataset

import pickle
import os
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
parser.add_argument('-b', '--batch', help="Batch size", type=int, default=30)
parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
parser.add_argument('-m', '--model', help="Model dir", required=True)
parser.add_argument('-c', '--checkpoint', help="Old Model to load", default = "NoneExist")

splt = ""
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
    print("=========SubToSub=========")
    #load data
    datasets = DramaDataset(basedir=DataDir,
                            maxFrame=0,
                            maxSeries=5,
                            )
    loader = torch.utils.data.DataLoader(datasets, batch_size=BatchSize, shuffle=True, num_workers=2)
    
    lang, model = loadModel(modelDir=modelDir,
                            LangFile=LangFile,
                            modelFile=modelFile,
                            dataset=datasets)
    
    print("Data size\t: {}".format(len(datasets)))
    print("Max epoch\t: {}\nBatch size\t: {}\nLearning rate\t: {}\n".format(MaxEpoch, BatchSize, lr))
    print("Start training........\n")
    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    writer = SummaryWriter(modelDir)
    recLoss = Average()
    timer = Timer()
    trainStep = 0
    
    for epoch in range(MaxEpoch):
        for i, data in enumerate(loader, 1):
            try:
                pre, nex, imgs = data

                in_pre, in_nex, out_nex = transData(pre, nex, lang)

                outputs, hidden = model(in_pre, in_nex)

                outputs = flatMutileLength(outputs, out_nex[1])
                out_nexs = flatMutileLength(out_nex[0], out_nex[1])
                loss = criterion(outputs, out_nexs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                recLoss.addData(loss.item())
                writer.add_scalar('loss', loss.item(), trainStep)
                trainStep += 1
                if i % 100 == 0:
                    model.eval()
                    print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                                  .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
                    print("F: {}\nS: {}\nP: {}\n"
                          .format(pre[0], nex[0], predit(model, lang, pre[0])))
                    model.train()
            except Exception as exp:
                print(exp)
        if i % 100 != 0:
            print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                  .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))

        modelName = os.path.join(modelDir, "SubSubModel.{}.pth".format(int((epoch+1)/5)))
        print("Saving Epoch model: {}.....\n".format(modelName))
        torch.save(model, modelName)
    
def predit(model, lang, in_sents, max_length=50):
    ans = []
    in_seq = torch.LongTensor(lang.sentenceToVector(in_sents, sos=False, eos=False)).unsqueeze(0).to(device)
    inputs = torch.LongTensor([[lang["SOS"]]]).to(device)
    hidden = None
    
    cxt = model.makeContext(in_seq)
    for i in range(max_length):
        outputs, hidden = model.decode(inputs, cxt, hidden)
        prob, outputs = outputs.topk(1)

        if(outputs.item() == lang["EOS"]):
            break
        ans.append(outputs.item())
        inputs = outputs.squeeze(1).detach()
    return lang.vectorToSentence(ans)

def transData(in_sents, target_sents, lang):
    in_seqs = []
    in_targets = []
    out_targets = []
    
    vectorTransforms = [lambda x: torch.LongTensor(x).to(device)]
    
    for sent in in_sents:
        in_seqs.append(lang.sentenceToVector(sent, sos=False, eos=False))
    in_seqs = padding(in_seqs, lang["PAD"], vectorTransforms)
    
    for sent in target_sents:
        in_targets.append(lang.sentenceToVector(sent, sos=True, eos=False))
        out_targets.append(lang.sentenceToVector(sent, sos=False, eos=True))
    in_targets = padding(in_targets, lang["PAD"], vectorTransforms)
    out_targets = padding(out_targets, lang["PAD"], vectorTransforms)
    return in_seqs, in_targets, out_targets

def createLang(name, dataset):
    lang = Lang(name, splt)
    for data in dataset:
        lang.addSentance(data[0])
        lang.addSentance(data[1])
    print("Create lang model. Number of word: {}".format(len(lang)))
    
    return lang

def loadModel(modelDir, LangFile, modelFile, dataset):
    if not os.path.isdir(modelDir):
        os.mkdir(modelDir)

    if os.path.isfile(LangFile):
        with open(LangFile, 'rb') as f:
            lang = pickle.load(f)
            print("Load lang model: {}. Word size: {}".format(LangFile, len(lang)))
    else:
        lang = createLang(LangFile, dataset)
        with open(LangFile, 'wb') as f:
            pickle.dump(lang, f)
          
    if os.path.isfile(modelFile):
        model = torch.load(modelFile)
        print("Load model: {}.".format(modelFile))
    else:
        subencoderOpt = {
            "word_size": len(lang),
            "em_size": 512,
            "num_layers": 1,
            "dropout": 0.1,
            "hidden_size": 512,
            "output_size": 512 
        }
        decoderOpt = {
            "word_size": len(lang),
            "em_size": 512,
            "num_layers": 1,
            "dropout": 0.1,
            "hidden_size": 512,
            "feature_size": 512 
        }
        model = SubToSeq(subencoderOpt, decoderOpt)
        
    return lang, model