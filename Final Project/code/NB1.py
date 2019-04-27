#!/usr/bin/env python
# coding: utf-8
import re
import pandas as pd
import jieba
import random
from sklearn.model_selection import train_test_split

def getwords(doc):
    words = doc.split(' ')
    words = [word for word in words if word.isalpha()]
    with open(r'./stopwords.txt') as f:
        stopwords = f.read()
    stopwords = stopwords.split('\n')
    stopwords = set(stopwords)
    words = [word for word in words if word not in stopwords]
    return set(words)

class classifier:
    def __init__(self, getfeatures):
        self.fc = {}
        self.cc = {}
        self.getfeatures = getfeatures
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return self.fc[f][cat]
        else:
            return 0.0

    def catcount(self, cat):
        if cat in self.cc:
            return self.cc[cat]
        return 0

    def totalcount(self):
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()
    def train(self, item, cat):
        features = self.getfeatures(item)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0

        return self.fcount(f, cat)/self.catcount(cat)
    def weightedprob(self, f, cat, weight=1, ap=0.5):

        basicprob = self.fprob(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        bp = ((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp

class naivebayes(classifier):

    def __init__(self, getfeatures):
        classifier.__init__(self, getfeatures)

    def docprob(self, item, cat):
        features = self.getfeatures(item)
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat)
        return p
    def prob(self, item, cat):
        catprob = self.catcount(cat)/self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob
    def classify(self, item):
        max_prob = 0.0
        probs = {}
        for cat in range(4):
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max_prob:
                max_prob = probs[cat]
                best = cat
        return best
def totaltrain(cl,sample,label):
    if len(sample) != len(label):
        print('The length of sample and label are not the same!')
        return
    for i in sample['split'].keys():
        cl.train(sample['split'][i],label[i])
    print('train finished')

def acc(cl,X_test,Y_test):
    correct = 0
    case = 0
    for i in X_test['split'].keys():
        if cl.classify(X_test['split'][i]) == Y_test[i]:
            correct += 1
        case += 1
    # print('ACC:',correct/case)
    return correct/case

dataset = pd.read_csv("./data.csv",encoding = 'gbk')
dataset['split'] = dataset['Message'].apply(lambda x :' '.join(jieba.cut(x)))
X = dataset[['split','Len']]
Y = dataset['Label']

dataset2 = pd.read_csv("./data2.csv",encoding = 'gbk')
dataset2['split'] = dataset2['Message'].apply(lambda x :' '.join(jieba.cut(x)))
X2 = dataset2[['split','Len']]
Y2 = dataset2['Label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1,random_state = 10)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size = 0.1,random_state = 10)

cl1 = naivebayes(getwords)
totaltrain(cl1,X_train,Y_train)
print('ACC: ',acc(cl1,X_test,Y_test))

cl2 = naivebayes(getwords)
totaltrain(cl2,X_train2,Y_train2)
print('ACC2: ',acc(cl2,X_test2,Y_test2))







