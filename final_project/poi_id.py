#!/usr/bin/python

import sys
import pickle
sys.path.append("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/tools/")
sys.path.append("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/final_project/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus', 'salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list)



######## my trial

features_list = ['poi','bonus', 'deferral_payments', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'long_term_incentive', 'restricted_stock', 'restricted_stock_deferred',
                 'salary', 'shared_receipt_with_poi', 'to_messages', 
                 'total_payments', 'total_stock_value']

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels = data[:, 0]
features = data[:, 1:]

# pre-process dataset with z-scaling
from sklearn.preprocessing import scale
features = scale(features)

# pre-process dataset with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca.fit(features)
# print pca.explained_variance_ratio_
# the first seven components explained 59%, 21%, 6%, 5%, 3% and 2% variances
features = pca.fit_transform(features)

# split traning and testing dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# train various classifiers and test them

## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_GNB = GaussianNB()
## not bad

## Support vector machine
from sklearn.svm import SVC
clf_svm = SVC(C = 5, kernel = "poly")
## very poor

## Decision tree
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(max_depth = 3, min_samples_split = 10)
## soso

## Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf_adaboost = AdaBoostClassifier()
## seems overfitting

## Random forest
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()

## KNN
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors = 5)
# pretty good (but low precision)

## Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()

clf = clf_lr
# train the classifier and fir the model
clf.fit(features_train, labels_train)
# perform poor

# model evaluation
accuracy = clf.score(features_test, labels_test)
print "accuracy:", accuracy 
print "baseline accuracy:", float(sum(labels_test == 0))/len(labels_test)
test_prediction = clf.predict(features_test)
from sklearn.metrics import precision_score
print "precision:", precision_score(labels_test, test_prediction)
from sklearn.metrics import recall_score
print "precision:", recall_score(labels_test, test_prediction)
