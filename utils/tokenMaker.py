import jieba
class Lang:
    def __init__(self, name, split=""):
        self.name = name
        self.split = split
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        
        self.addWord("PAD")
        self.addWord("SOS")
        self.addWord("EOS")
        self.addWord("UNK")
        
    def  __getitem__(self, key):
        if isinstance(key, str):
            if key in self.word2index:
                return self.word2index[key]
            else:
                return self.word2index["UNK"]
        elif isinstance(key, int):
            if key < self.n_words:
                return self.index2word[key]
        return None
    def __len__(self):
        return self.n_words
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def __splitSentence(self, s):
        if self.split == "":
            return s
        else:
            return s.split(self.split)
    def addSentance(self, sent):
        for w in self.__splitSentence(sent):
            self.addWord(w)
    def sentenceToVector(self, s, sos = False, eos = False):
        numS = []
        if sos: 
            numS.append(self.word2index["SOS"])
        for w in self.__splitSentence(s):
            if w in self.word2index:
                numS.append(self.word2index[w])
            else:
                numS.append(self.word2index["UNK"])
        if eos:
            numS.append(self.word2index["EOS"])
        return numS
    def vectorToSentence(self, v):
        if self.word2index["SOS"] in v:
            v.remove(self.word2index["SOS"])
        if self.word2index["EOS"] in v:
            v.remove(self.word2index["EOS"])

        s = self.split.join(self.index2word[i] for i in v)
        return s
    
class LangV2:
    def __init__(self, name, split="jieba"):
        self.name = name
        self.split = split
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        
        self.addWord("PAD")
        self.addWord("SOS")
        self.addWord("EOS")
        self.addWord("UNK")
        
    def  __getitem__(self, key):
        if isinstance(key, str):
            if key in self.word2index:
                return self.word2index[key]
            else:
                return self.word2index["UNK"]
        elif isinstance(key, int):
            if key < self.n_words:
                return self.index2word[key]
        return None
    def __len__(self):
        return self.n_words
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def splitSentence(self, s):
        if self.split == "":
            return s
        if self.split == "jieba":
            return jieba.cut(s)
        else:
            return s.split(self.split)
    def addSentance(self, sent):
        for w in self.splitSentence(sent):
            self.addWord(w)
    def sentenceToVector(self, s, sos = False, eos = False):
        numS = []
        if sos: 
            numS.append(self.word2index["SOS"])
        for w in self.splitSentence(s):
            if w in self.word2index:
                numS.append(self.word2index[w])
            else:
                numS.append(self.word2index["UNK"])
        if eos:
            numS.append(self.word2index["EOS"])
        return numS
    def vectorToSentence(self, v):
        if self.word2index["SOS"] in v:
            v.remove(self.word2index["SOS"])
        if self.word2index["EOS"] in v:
            v.remove(self.word2index["EOS"])
        sp = " " if self.split == " " else ""
        s = sp.join(self.index2word[i] for i in v)
        return s