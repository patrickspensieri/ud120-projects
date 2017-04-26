#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "num people : ", len(enron_data)
print "num features per person : ", len(enron_data.itervalues().next())
numPoi = 0
numSalary = 0
numEmail = 0
numTotalPayments = 0
numPoiWithNaNTtlPayments = 0
for dict in enron_data.itervalues():
    if dict["poi"] == 1:
        numPoi += 1
        if dict["total_payments"] == "NaN":
            numPoiWithNaNTtlPayments += 1
    if dict["salary"] != "NaN":
        numSalary += 1
    if dict["email_address"] != "NaN":
        numEmail += 1
    if dict["total_payments"] != "NaN":
        numTotalPayments += 1
print "num poi : ", numPoi
print "num salary : ", numSalary
print "num email_address : ", numEmail
print "James Prentice total_stock_value : ", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "num NaN total_payments", len(enron_data) - numTotalPayments
print "num NaN total_payments AND poi: ", numPoiWithNaNTtlPayments


