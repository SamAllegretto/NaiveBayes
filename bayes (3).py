#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
from math import log


class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = {}
        self._train_len = 0.0

    def train(self, X, y):
        
        self._Ncls = [0] * len(set(y))
        
        for i in set(y):
            self._Nfeat[i] = {}
            
        
        for ins, clas in zip(X, y):           
            self._Ncls[clas] += 1.0
            for attr in range(len(ins)):
                if attr in self._Nfeat[clas]:
                    self._Nfeat[clas][attr] += float(ins[attr])
                else:
                    self._Nfeat[clas][attr] = 0.0
                
        return 

    def predict(self, X):
       
        predictions = [0] * len(X)
        
        for i, X_instances in enumerate(X):

            decision = None
            top_prob = -float('inf')
            

            for clas in range(len(self._Ncls)):
                
                total_prob = 0.0
                curr_class = 0.0
                total_class = 0.0

                for j in range(len(self._Ncls)):
                    total_class += self._Ncls[j]
            

                curr_class = self._Ncls[clas] + self._smooth
                total_class = total_class + (len(self._Ncls) * self._smooth)
                class_prob = (curr_class/total_class)
                total_prob = log(class_prob)

                for attr in range(len(X_instances)):

                    total_attr = self._Ncls[clas] + (self._smooth * 2.0)
                    curr_attr = self._Nfeat[clas][attr] + self._smooth 
                    prob_attr = curr_attr/total_attr
                    total_prob += X_instances[attr] * log(prob_attr)

                y_hat = total_prob
                #print(y_hat)
        
                if y_hat > top_prob:
                    top_prob = y_hat
                    decision = clas

            #print (decision)
            predictions[i] = decision

        return predictions
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        #return np.zeros([X.shape[0],1])

class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        # Your code goes here.
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])
        


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=True)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
print ('alpha=%f accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))


