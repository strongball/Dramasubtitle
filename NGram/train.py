import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.utils.data

from utils.tokenMaker import Lang
from utils.tool import Timer, Average

from dataset.readVideo import DramaDataset

from NGram.tool import NGram, trigrams
import pickle
import os

def trainer(args):
    modelDir = args.model
    LangFile = os.path.join(modelDir, "Lang.pkl")
    modelFile = args.checkpoint
    
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
    
    lang, model = loadmodel(modelDir=modelDir,
                            LangFile=LangFile,
                            modelFile=modelFile,
                            dataset=datasets)
    
    print("Data size\t: {}".format(len(datasets)))
    print("Max epoch\t: {}\nBatch size\t: {}\nLearning rate\t: {}\n".format(MaxEpoch, BatchSize, lr))
    print("Start training........\n")
    
    if use_cuda:
        global Variable
        Variable = Variable.cuda
        model.cuda()
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    recLoss = Average()
    timer = Timer()
    
    for epoch in range(MaxEpoch):
        for i, data in enumerate(loader, 1):
            pre,nex,_ = data
            for sentence in pre+nex:
                for context, target in trigrams(lang.sentenceToVector(sentence, sos=False, eos=False)):
                    
                    context_var = Variable(torch.LongTensor(context))
                    model.zero_grad()
                    log_probs = model(context_var)
                    loss = criterion(log_probs, Variable(torch.LongTensor([target])))
                    loss.backward()
                    optimizer.step()
                    recLoss.addData(loss.data[0])
                    
            if i % 50 == 0:
                print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                      .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
        if i % 50 != 0:
            print("Epoch: {:2d}, Step: {:5d}, Time: {:6.3f}, Loss: {:7.5f}"
                  .format(epoch, i, timer.getAndReset(), recLoss.getAndReset()))
            
        modelName = os.path.join(modelDir, "NGram.{}.pkl".format(int((epoch+1)*10/MaxEpoch)))
        print("Saving Epoch model: {}.....\n".format(modelName))
        torch.save(model, modelName)
    
def createLang(name, dataset):
    lang = Lang(name, " ")
    for data in dataset:
        lang.addSentance(data[0])
        lang.addSentance(data[1])
    print("Create lang model. Number of word: {}".format(len(lang)))
    
    return lang

def loadmodel(modelDir, LangFile, modelFile, dataset):
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
        model = NGram(len(lang), 256, 2)
        
    return lang, model