#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:27:01 2023

@author: tremblayju
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naive_bayes_v1 import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Import data:


print(X_train.shape)
print(y_train.shape)
print(X_train[:5, :])

nb = NaiveBayes()
# train data
nb.fit(X_train, y_train)
# And then predict outcome of test data using training model.
predictions = nb.predict(X_test)
print(accuracy(y_test, np.array(predictions)))
