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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()
#reduce size of training data to 1/100 of original, lower precision
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

#num columns = num features
#num features can be tuned in email_preprocess.py,
#by lowering percentile param in SelectPercentile
print "number of features : ", len(features_test[0])

#code starts here
clf = DecisionTreeClassifier(min_samples_split=40)

#train the classifier
t0 = time()
clf.fit(features_train, labels_train)
print "training time : ", round(time() - t0, 3), "s"

#predict new labels
t0 = time()
pred = clf.predict(features_test)
print "prediction time : ", round(time() - t0), "s"
print accuracy_score(labels_test, pred)



