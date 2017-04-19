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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#code starts here
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#params     kernal - map into higher dimensional space to find linear separation, go back to original space
#           C - tradeoff between smooth decision boundary, and classifying training points correctly
#           gamma - higher values prioritize points closer to decision boundary


# clf = SVC(kernel='linear')              #linear
clf = SVC(C = 10000, kernel = "rbf")    #optimized rbf

#reduce size of training data to 1/100 of original, lower precision
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

#train the classifier
t0 = time()
clf.fit(features_train, labels_train)
print "training time : ", round(time() - t0, 3), "s"

#predict new labels
t0 = time()
pred = clf.predict(features_test)
print "prediction time : ", round(time() - t0, 3), "s"

#score the prediction
score = accuracy_score(labels_test, pred)
print score

print "pred.size : ", pred.size
print "pred.sum() : ", pred.sum()



