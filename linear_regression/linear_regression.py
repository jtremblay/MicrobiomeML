#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:28:43 2023

@author: tremblayju
"""
import numpy as np

class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
        self.bias_list = []
        self.weight_list = []
        self.dw_list = []
        self.db_list = []
        # square residuals.
        self.w_res_list = [] # will store weight for each iteration
        self.b_res_list = [] # will store bias for each iteration
        
        
    def fit(self, X, y):
        # init parameters
        # if 1 dimension, weight will be shape n, 1, two dimension: n,2, etc.
        # starting weight and bias = 0
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features) 
        #print("DEBUG: self.weight -> " + str(self.weight))
        self.bias = 0
        
        for _ in range(self.n_iters):
            #print("DEBUG: X -> " + str(X)) # always the same X vector
            #print("DEBUG: self.weight -> " + str(self.weight))
            #print("DEBUG: self.bias -> " + str(self.bias))
            
            # y_predicted is a vector with 80 el. no.dot multiply each elements
            # of the vector by the self.weights value
            y_predicted = np.dot(X, self.weight) + self.bias
            #print("DEBUG: y_predicted.shape -> " + str(y_predicted.shape))
            
            #print("DEBUG: X.T.shape" + str(X.T.shape))
            #print("DEBUG: y_predicted - y .shape -> " + str((y_predicted - y).shape)) 
            #print("DEBUG: X.T -> " + str(X.T)) 
            #print("DEBUG: y_predicted - y -> " + str(y_predicted - y)) 
            #tmp = np.dot(X.T, (y_predicted - y))
            #print(tmp)
            # Here np.dot will multipley each X[n] by corresponding y[n] and sum all the results.
            sum_square_res_w = np.dot(X.T, (y_predicted - y)**2)
            sum_square_res_b = np.sum((y_predicted - y)**2)
            #print("DEBUG: sum_square_res_w -> " + str(sum_square_res_w[0]))
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            #print("DEBUG: dw -> " + str(dw))
            #print("DEBUG: db -> " + str(db))
            
            # update weight and bias. learning_rate * derivative
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
            #print("DEBUG: self.bias -> " + str(self.bias))
            
            # for each iteration populate results for diagnosis
            self.bias_list.append(self.bias)
            self.weight_list.append(self.weight[0])
            self.dw_list.append(dw)
            self.db_list.append(db)
            self.w_res_list.append(sum_square_res_w[0])
            self.b_res_list.append(sum_square_res_b)
    
    # Once fit is done, test model with test data using obtained weight and bias
    def predict(self, X):
        y_predicted = np.dot(X, self.weight) + self.bias
        return y_predicted