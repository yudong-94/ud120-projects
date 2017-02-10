#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

### knn
from sklearn.neighbors import KNeighborsClassifier

n_nlists = [n for n in range(1,51) if n%5 == 0]

for n_neighbors in n_nlists:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(features_train, labels_train) 
    print "n_neighbors is ", n_neighbors
    print "score is ", clf.score(features_test, labels_test)

# change the number of neighbors: 0.92 - 0.936 - 0.928, highest as 0.936 （n=20)

clf = KNeighborsClassifier(n_neighbors=5, weights = "distance")
clf.fit(features_train, labels_train) 
print "score is ", clf.score(features_test, labels_test)

# change weights to "distance": increase the accuracy rate

### adaboost

from sklearn.ensemble import AdaBoostClassifier

n_nlists = [n for n in range(1,101) if n%10 == 0]

for n_estimators in n_nlists:
    clf = AdaBoostClassifier(n_estimators = n_estimators)
    clf.fit(features_train, labels_train) 
    print "n_estimator is ", n_estimators
    print "score is ", clf.score(features_test, labels_test)
    
# change the number of estimators: 0.916-0.928-0.924 (highest at n = 20)

for learning_rate in range(1, 6):
    clf = AdaBoostClassifier(learning_rate = learning_rate)
    clf.fit(features_train, labels_train) 
    print "learning_rate is ", learning_rate
    print "score is ", clf.score(features_test, labels_test)
    
# increase the learning rate: 0.924 - 0.764, always decrease


### random forest
from sklearn.ensemble import RandomForestClassifier

n_nlists = [n for n in range(1,101) if n%10 == 0]

for n_estimators in n_nlists:
    clf = RandomForestClassifier(n_estimators = n_estimators)
    clf.fit(features_train, labels_train) 
    print "n_estimator is ", n_estimators
    print "score is ", clf.score(features_test, labels_test)

# change the number of estimators: 0.92-0.928-0.908 (highest at n = 30)

for max_depth in range(1,11):
    clf = RandomForestClassifier(max_depth = max_depth)
    clf.fit(features_train, labels_train) 
    print "max_depth is ", max_depth
    print "score is ", clf.score(features_test, labels_test)

# change the max depth: 0.804 - 0.924 - 0.916 (highest at n = 3, 4)

for min_samples_split in range(2, 21, 2):
    clf = RandomForestClassifier(min_samples_split = min_samples_split)
    clf.fit(features_train, labels_train) 
    print "min_samples_split is ", min_samples_split
    print "score is ", clf.score(features_test, labels_test)

# change the min_samples_split: 0.92 - 0.932 - 0.916 (highest at n = 4)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
