#!/usr/bin/python

from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


#start knn
knn = KNeighborsClassifier(weights="distance", n_neighbors=10)
t0 = time()
knn.fit(features_train, labels_train)
print "training knn : ", round(time() - t0), "s"
t0 = time()
predKnn = knn.predict(features_test)
print "prediction knn : ", round(time() - t0), "s"
scoreKnn = accuracy_score(labels_test, predKnn)
print "score knn : ", scoreKnn
#end knn

#start random forest
rf = RandomForestClassifier(n_estimators=10, min_samples_split=20)
t0 = time()
rf.fit(features_train, labels_train)
print "training rf : ", round(time() - t0), "s"
t0 = time()
predRf = rf.predict(features_test)
print "prediction rf : ", round(time() - t0), "s"
scoreRf = accuracy_score(labels_test, predRf)
print "score rf : ", scoreRf
#end random forest

clf = knn


# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
