# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    #print("DEBUG: input vector 1 => " + str(x1))
    #print("DEBUG: input vector 2 => " + str(x2))
    #my_sub = x1 - x2
    #my_sqr = my_sub**2
    #print("DEBUG: my_sub -> " + str(my_sub))
    #print("DEBUG: my_sqr -> " + str(my_sqr))
    #my_sum = np.sum(my_sqr)
    #print("DEBUG: my_sum =>" + str(my_sum))
    #my_sqr2 = np.sqrt(my_sum)
    #print("DEBUG: my_sqr2 -> " + str(my_sqr2))
    res = np.sqrt(np.sum((x1-x2)**2))
    #print("DEBUG: distances => " + str(res))
    
    #DEBUG: input vector 1 => [4.8 3.4 1.6 0.2]
    #DEBUG: input vector 2 => [5.8 2.7 4.1 1. ]
    #DEBUG: my_sub -> [-1.   0.7 -2.5 -0.8]
    #DEBUG: my_x2 -> [1.   0.49 6.25 0.64]
    #DEBUG: (np.sum, axis=none) my_sum =>8.379999999999997
    #DEBUG: my_sqr2 -> 2.8948229652260253
    #DEBUG: distances => 2.8948229652260253
    
    
    return res
    

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X): # X is our test dataset (the smaller 30 elements one)
        #print("DEBUG:" + str(X))
        # one by one loop through each row of the nd array.
        # So basically, for each 4 dimension test vector, compute distance 
        # against all other vectors in the training dataset X_train, then
        # take the most k common samples and perform a simple majority vote.
        predicted_labels = [self._predict(x) for x in X] #return list
        return np.array(predicted_labels)
    
    def _predict(self, x):
        # compute distances
        print("DEBUG: vector to evaluate =>  " + str(x) )
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #print("DEBUG: distances => " + str(distances))
        # get k nearest samples, labels too
        k_indices = np.argsort(distances)[:self.k]
        print("DEBUG: k_indices => " + str(k_indices)) # get the indices of the k closest samples.
        # Then, we go get into the y_train (the actual results) the most closest k classification values (0, 1 or 2)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print("DEBUG: k_nearest_labels -> " + str(k_nearest_labels))
        # majority vote, most common class label
        # once we have these k most closest 5 results, do a majority vote with most_common() (Counter library).
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        
        
    
    