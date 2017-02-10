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

enron_data = pickle.load(open("/Users/hzdy1994/Desktop/Machine Learning/Udacity/ud120-projects/final_project/final_project_dataset.pkl", "r"))

# number of people
print len(enron_data)

# number of features for each person
print len(enron_data["ALLEN PHILLIP K"])

# number of POIs
no_poi = 0
for person in enron_data:
    if enron_data[person]["poi"]:
        no_poi += 1
print no_poi

# total stock belongs to James Prentice
print enron_data["PRENTICE JAMES"]["total_stock_value"] 

# email messages from Wesley Colwell to persons of interest
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# stock option exercised by Jeffrey K Skilling
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# among CEO, CFO, Chairman, who got the most money
for people in ["SKILLING JEFFREY K", "FASTOW ANDREW S", "LAY KENNETH L"]:
    print people, enron_data[people]["total_payments"]
    
# how many poeple have a qualified salary; how many for known email address
valid_salary = 0
valid_email = 0
for person in enron_data:
    if enron_data[person]["salary"] != "NaN":
        valid_salary += 1
    if enron_data[person]["email_address"] != "NaN":
        valid_email += 1
print valid_salary, valid_email

# how many percentage of people don't have total payments data
missing_payments = 0
for person in enron_data:
    if enron_data[person]["total_payments"] == "NaN":
        missing_payments += 1

print float(missing_payments)/len(enron_data)

# how many POIs don't have total payments data
missing_payments_poi = 0
for person in enron_data:
    if enron_data[person]["total_payments"] == "NaN" and enron_data[person]["poi"]:
        missing_payments_poi += 1

print missing_payments_poi