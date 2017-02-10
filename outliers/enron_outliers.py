#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

data_dict.pop( "TOTAL", 0 )

data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()