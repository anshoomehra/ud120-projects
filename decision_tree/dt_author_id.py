#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import sys

import numpy as np

# sklearn accuracy_score function
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    return clf.fit(features_train,labels_train)

print "Number of Features",len(features_train[0])

clf = classify(features_train, labels_train)
pred = clf.predict(features_test)

print "Accuracy:", accuracy_score(pred, labels_test)
#########################################################


