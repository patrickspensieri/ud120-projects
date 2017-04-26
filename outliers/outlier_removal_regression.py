#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle
from outlier_cleaner import outlierCleaner
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )

# ages and net_worths need to be reshaped into 2D numpy arrays
# second argument of reshape command is a tuple of integers: (n_rows, n_columns)
# by convention, n_rows is the number of data points
# and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

# reg before cleaning
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
print "coef_ before cleaning : ", reg.coef_
print "score before cleaning : ",  reg.score(ages_test, net_worths_test)
print "type() ages_train : ", type(ages_train)

# try:
#     plt.plot(ages, reg.predict(ages), color="blue")
# except NameError:
#     pass
# plt.scatter(ages, net_worths)
# plt.show()

# identify and remove the most outlier-y points
# predictions are the predicted targets from regression
# ages is list of ages in training set
# net_worths is actual value of net_worths in the training set
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    print predictions
    cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train)
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"

# only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    # refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print "coef_ AFTER cleaning : ", reg.coef_
        print "score AFTER cleaning : ", reg.score(ages_test, net_worths_test)
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"

