#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
# sklearn accuracy_score function
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#########################################################

## Trim down training data set to just 1% of total data for trying impact on accuracy & run time ..
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

# Instantiate SVC Classifier with Linear Kernel
# Accuracy 98% on all data samples (training/prediction time  3 min/16 sec), and 88% on just 1% of training data (training, prediction time in sub-secs)
#clf = SVC(kernel="linear")

# Instantiate SVC Classifier with RBF Kernel
# Accuracy 62% on just 1% of training data (training/prediction times as sub-secs/1sec)
#clf = SVC(kernel="rbf")

# Instantiate SVC Classifier with RBF Kernel
# Accuracy 62% on just 1% of training data (training/prediction times as sub-secs/1sec)
#clf = SVC(kernel="rbf", C=10.)

# Instantiate SVC Classifier with RBF Kernel
# Accuracy 62% on just 1% of training data (training/prediction times as sub-secs/1sec)
#clf = SVC(kernel="rbf", C=100.)

# Instantiate SVC Classifier with RBF Kernel
# Accuracy 82% on just 1% of training data (training/prediction times as sub-secs)
#clf = SVC(kernel="rbf", C=1000.)

# Instantiate SVC Classifier with RBF Kernel
# Accuracy 99% on all training data (training/prediction times as 2 minutes / 10 sec)
# Accuracy 89% on just 1% of training data (training/prediction times as sub-secs)
clf = SVC(kernel="rbf", C=10000.)

## Time track to train
t0 = time()

# Train the Classifier
clf.fit(features_train, labels_train)

print "Training Time:", round(time()-t0, 3), "s"

## Time track to predict
t0 = time()

# Prediction on test data
#pred = clf.predict(features_test[50].reshape(1,-1))
pred = clf.predict(features_test)

#print ("Prediction :", pred)

print "Prediction Time:", round(time()-t0, 3), "s"

# Overall Accuracy
accuracy_score = accuracy_score(pred, labels_test)

print ("Accuracy Score: ", accuracy_score)
