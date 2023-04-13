#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:51:19 2023

@author: tremblayju
"""

import numpy as np
import sys
import logging

class NaiveBayes:
   
    def __init__(self, verbose=False):
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # y is the actual answer.
        n_classes = len(self._classes)
        
        self._logger.debug('Classes:')
        self._logger.debug(self._classes)
        
        # init mean, var, prior
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # mean for each feature
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) # var for each feature
        self._priors = np.zeros(n_classes, dtype=np.float64) # global prob for each class.
        #print("self._mean:")
        #print(self._mean)
        #print("self._var:")
        #print(self._var)
        #print("self.priors:")
        #print(self._priors)
        #print("X:")
        #print(X)
        #print(y)
        # So basically, here we'll loop through our 2 classes (0 or 1) - which again are the answers
        # So one row per answer. Remember that our self._mean array is 2 (because two classes or types of answers) 
        # x 10 (reps i.e. samples). Here we will just populate this array with the mean values for each class/answer
        for c in self._classes: 
            # here c (i.e answer or classes) is an int, might be more problematic if string.
            # In other words c work because it is either 0 or 1 which can be easily used 
            # as an array index.
            X_c = X[c==y] # select by row based on another array.
                          # here we'll have a n classes row x 10 col array
                          # and then we'll compute the mean for current class
            self._logger.debug("X_c.dtype:")
            self._logger.debug(X_c.dtype)
            # In this next line, we actually compute the mean of the X_c array and store it  
            # in the index #c(i.e. 0 or 1) of the self._mean array. 
            # again one index per class
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

        self._logger.debug("self._mean:")
        self._logger.debug(self._mean)
        self._logger.debug(self._mean.shape)
        self._logger.debug("self._var:")
        self._logger.debug(self._var)
        self._logger.debug(self._var.shape)
        self._logger.debug("self._priors:")
        self._logger.debug(self._priors)
        self._logger.debug(self._priors.shape)
        
    # Then, predict.
    def predict(self, X):
        y_pred = [self._predict(x) for x in X] # do _predict for each row/array of 10 elements
        return y_pred
    
    def _predict(self, x):
        self._logger.debug("x in _predict:")
        self._logger.debug(x)
        posteriors = [] # will ultimately contain 2 values, one for each class. 
                        # will then return the highest values between the 2 classes.
        # loop through the 2 classes/answers
        # Here, enumerate return index and the actual value of the self._classes array.
        # So here in this loop we extract the prior, class, posterior, etc
        #print("-------------------------")
        for idx, c in enumerate(self._classes):
            self._logger.debug("idx: " + str(idx) + "  c: " + str(c))
            self._logger.debug(self._priors[idx])
            prior = np.log(self._priors[idx]) #get prior for curr class
            self._logger.debug(prior)
            class_conditional =  np.sum(np.log(self._pdf(idx, x))) # compute pdf for curr 10 elements.
            posterior = prior + class_conditional
            posteriors.append(posterior) # Here we litteraly append the two answers.
        
        self._logger.debug("posteriors")
        self._logger.debug(posteriors)
        #print(self._classes)
        return self._classes[np.argmax(posteriors)] # and here we return the best answer for  
                                                    # this current row.
    
    # probability density function (Gaussian)
    # http://www.saedsayad.com/naive_bayesian.htm
    # for each 1d array of 10 values, get mean, var of their associated class
    # and compute numerator and denominator and return n/d
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx] # get mean of curr class
        var = self._var[class_idx]   # get variance of curr class
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
            
