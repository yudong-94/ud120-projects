#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    remove_num = int(0.1 * len(predictions))
    errors = predictions - net_worths
    topn = np.copy(errors)
    topn.sort(axis = 0)
    topn = topn[len(predictions)-remove_num:]
    
    for index in xrange(len(predictions)):
        if errors[index] not in topn:
            cleaned_data.append((ages[index], net_worths[index], errors[index]))

    return cleaned_data

