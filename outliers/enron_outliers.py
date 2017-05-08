#!/usr/bin/python

import pickle
import sys
import math
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r"))
# remove outlier, TOTAL column from spreadsheet
data_dict.pop("TOTAL", None)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
print "type(data_dict) : ", type(data_dict)
print "type(data) : ", type(data)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
for person in data_dict:
    if type(data_dict[person]["salary"]) == int and type(data_dict[person]["bonus"]) == int:
        if data_dict[person]["bonus"] > 5000000 and data_dict[person]["salary"] > 1000000:
            print data_dict[person]["salary"], " ", data_dict[person]["bonus"], " ", person

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
