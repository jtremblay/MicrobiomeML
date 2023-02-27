#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:35:52 2023

@author: tremblayju
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1234)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#plt.figure()
#plt.scatter(X[:,0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
#plt.show()

#a = [10,10,10,10,2,2,3,4,5,6]
#from collections import Counter
#most_common = Counter(a).most_common(1)
#print(most_common)

from knn import KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# assume in the same order...?
print(predictions)
print(y_test)
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

print(y_train)