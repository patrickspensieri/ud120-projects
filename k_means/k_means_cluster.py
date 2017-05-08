#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# featureScaling normalizes each value as a num between 0 and 1
# note that sklearn already implements featureScaling, which will be used above
def featureScaling(arr):
    maxVal = arr[0]
    minVal = arr[0]
    for num in arr:
        maxVal = max(maxVal, num)
        minVal = min(minVal, num)
    # for num in arr:
    #     num = (num - minVal) / (maxVal - minVal)
    arr = [(num - minVal)/float(maxVal - minVal) for num in arr]
    return arr

def featureScalingTest(val, minVal,maxVal):
    return (val - minVal) / float(maxVal - minVal)

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2",):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    # plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
# get max and min values for exercised_stock_options, ignoring NaN
maxStock = 0
minStock = sys.maxint
maxSalary = 0
minSalary = sys.maxint
for person in data_dict:
    if type(data_dict[person]["exercised_stock_options"]) == int:
        maxStock = max(maxStock, data_dict[person]["exercised_stock_options"])
        minStock = min(minStock, data_dict[person]["exercised_stock_options"])
    if type(data_dict[person]["salary"]) == int:
        maxSalary = max(maxSalary, data_dict[person]["salary"])
        minSalary = min(minSalary, data_dict[person]["salary"])
print "max exercised_stock_options : ", maxStock
print "min exercised_stock_options : ", minStock
print "max salary : ", maxSalary
print "min salary : ", minSalary
# custom function, test for feature scaling
print "salary of 200 000 : ", featureScalingTest(200000, minSalary, maxSalary)
print "exercised_stock_options of 1 000 000 : ", featureScalingTest(1000000, minStock, maxStock)



### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
# feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
# plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
km = KMeans(n_clusters = 2)
pred = km.fit_predict(data)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

