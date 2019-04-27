#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

def lengthencode(x):
    if x==5:
        return 1
    elif x==11:
        return 2
    elif x==13:
        return 2
    return 0

dataset = pd.read_csv("./data.csv",encoding = 'gbk')
dataset['split'] = dataset['Message'].apply(lambda x :' '.join(jieba.cut(x)))
dataset['Len'] = dataset['Len'].apply(lambda x :lengthencode(x))
X = dataset[['split','Len']]
Y = dataset['Label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1,random_state = 10)

CountVectorizer = CountVectorizer()
TfidfTransformer = TfidfTransformer()
X_train_termcounts = CountVectorizer.fit_transform(X_train['split'])
X_test_termcounts = CountVectorizer.transform(X_test['split'])
X_train_tfidf = TfidfTransformer.fit_transform(X_train_termcounts)
X_test_tfidf = TfidfTransformer.transform(X_test_termcounts)

X_train_termcounts = np.column_stack((X_train_termcounts.toarray(),X_train['Len']))
X_train_tfidf = np.column_stack((X_train_tfidf.toarray(),X_train['Len']))

X_test_termcounts = np.column_stack((X_test_termcounts.toarray(),X_test['Len']))
X_test_tfidf = np.column_stack((X_test_tfidf.toarray(),X_test['Len']))

classifier = MultinomialNB().fit(X_test_termcounts,Y_train)
Y_predict = classifier.predict(X_test_tfidf)
accuracy_score(Y_test,Y_predict)

classifier = LogisticRegression().fit(X_test_termcounts,Y_train)
Y_predict = classifier.predict(X_test_tfidf)
accuracy_score(Y_test,Y_predict)

classifier = SVC().fit(X_test_termcounts,Y_train)
Y_predict = classifier.predict(X_test_tfidf)
accuracy_score(Y_test,Y_predict)

classifier = RandomForestClassifier().fit(X_test_termcounts,Y_train)
Y_predict = classifier.predict(X_test_tfidf)
accuracy_score(Y_test,Y_predict)

classifier = GradientBoostingClassifier().fit(X_test_termcounts,Y_train)
Y_predict = classifier.predict(X_test_tfidf)
accuracy_score(Y_test,Y_predict)
