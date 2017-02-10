#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

### train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

test_prediction = clf.predict(X_test)
print sum(test_prediction == 1)
print len(test_prediction)
print sum(y_test)

# confusion matrix
tp = 0

for i in range(len(test_prediction)):
    if test_prediction[i] == y_test[i] and y_test[i] == 1:
        tp += 1

from sklearn.metrics import precision_score
print precision_score(y_test, test_prediction)

from sklearn.metrics import recall_score
print recall_score(y_test, test_prediction)