import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.utils.data


from modal.BigModal import SubToSeq
from utils.tokenMaker import Lang
from utils.tool import padding, flatMutileLength, Timer, Average
from dataset.readVideo import DramaDataset

import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
parser.add_argument('-b', '--batch', help="Batch size", type=int, default=30)
parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
parser.add_argument('-m', '--modal', help="Model dir", required=True)
parser.add_argument('-c', '--checkpoint', help="Old Model to load", default = "NoneExist")

def trainer(args):
    ModalDir = args.modal
    LangFile = os.path.join(ModalDir, "Lang.pkl")
    ModalFile = args.checkpoint
    
    MaxEpoch = args.epoch
    BatchSize = args.batch
    DataDir = args.data
    lr = args.lr
    
    use_cuda = torch.cuda.is_available()
    
    #load data
    datasets = DramaDataset(basedir=DataDir,
                        maxFrame=0,
                        )
    loader = torch.utils.data.DataLoader(datasets, batch_size=BatchSize, shuffle=True, num_workers=2)
    
    lang, modal = loadModal(ModalDir=ModalDir,
                            LangFile=LangFile,
                            ModalFile=ModalFile,
                            dataset=datasets)
    
    print("Data size: {}".format(len(datasets)))
    print("Max epoch:{}\nBatch size:{}\nLearning rate:{}\n".format(MaxEpoch, BatchSize, lr))
    print("Start training........\n")
    
    if use_cuda:
        global Variable
        Variable = Variable.cuda
        modal.cuda()
    modal.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modal.parameters(), lr=lr)
    
    recLoss = Average()
    timer = Timer()
    for epoch in range(MaxEpoch):
        for i, data in enumerate(loader, 1):
            try:
                pre, nex, imgs = data

                in_pre, in_nex, out_nex = transData(pre, nex, lang)

                outputs, hidden = modal(in_pre, in_nex)

                outputs = flatMutileLength(outputs, out_nex[1])
                out_nexs = flatMutileLength(out_nex[0], out_nex[1])
                loss = criterion(outputs, out_nexs)
                recLoss.addData(loss.data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                                  .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
                    print("F: {}\nS: {}\nP: {}\n"
                          .format(pre[0], nex[0], predit(modal, lang, pre[0])))
            except Exception as exp:
                print(exp)
        print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                      .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))

        modalName = os.path.join(ModalDir, "SubSubModal.{}.pkl".format(int((epoch+1)*10/MaxEpoch)))
        print("Saving Epoch modal: {}.....\n".format(modalName))
        torch.save(modal, modalName)
    
def predit(modal, lang, in_sents, max_length=50):
    ans = []
    in_seq = Variable(torch.LongTensor(lang.sentenceToVector(in_sents, sos=False, eos=False)).unsqueeze(0))
    inputs = Variable(torch.LongTensor([[lang["SOS"]]]).long())
    hidden = None
    
    cxt = modal.makeContext(in_seq)
    for i in range(max_length):
        outputs, hidden = modal.decode(inputs, cxt, hidden)
        prob, outputs = outputs.topk(1)
        outputs = outputs[0][0].data[0]
        if(outputs == lang["EOS"]):
            break
        ans.append(outputs)
        inputs = Variable(torch.LongTensor([[outputs]]))
    return lang.vectorToSentence(ans)

def transData(in_sents, target_sents, lang):
    in_seqs = []
    in_targets = []
    out_targets = []
    
    vectorTransforms = [torch.LongTensor, Variable]
    
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
    lang = Lang(name, "")
    for data in dataset:
        lang.addSentance(data[0])
        lang.addSentance(data[1])
    print("Create lang model. Number of word: {}".format(len(lang)))
    
    return lang

def loadModal(ModalDir, LangFile, ModalFile, dataset):
    if not os.path.isdir(ModalDir):
        os.mkdir(ModalDir)

    if os.path.isfile(LangFile):
        with open(LangFile, 'rb') as f:
            lang = pickle.load(f)
            print("Load lang model: {}. Word size: {}".format(LangFile, len(lang)))
    else:
        lang = createLang(LangFile, dataset)
        with open(LangFile, 'wb') as f:
            pickle.dump(lang, f)
          
    if os.path.isfile(ModalFile):
        modal = torch.load(ModalFile)
        print("Load model: {}.".format(ModalFile))
    else:
        subencoderOpt = {
            "word_size": len(lang),
            "em_size": 512,
            "hidden_size": 512,
            "num_layers": 1,
            "output_size": 512 
        }
        decoderOpt = {
            "word_size": len(lang),
            "em_size": 512,
            "hidden_size": 512,
            "num_layers": 1,
            "feature_size": 512 
        }
        modal = SubToSeq(subencoderOpt, decoderOpt)
        
    return lang, modal