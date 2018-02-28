#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB
# sklearn accuracy_score function
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

## Instantiate GNB Classifier
clf = GaussianNB()

## Time track to train 
t0 = time()

## Traing Classifier for training dataset
clf.fit(features_train,labels_train)

print "Training Time:", round(time()-t0, 3), "s"

## Time track to predict 
t0 = time()

## Predictions on test dataset
pred = clf.predict(features_test)

print "Prediction Time:", round(time()-t0, 3), "s"

## Accuracy Score based on predictions and actual labels
acc_score = accuracy_score(pred, labels_test)

print("Accuracy Score is {}".format(acc_score))

#########################################################


