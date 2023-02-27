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

from naive_bayes_v2 import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

##########################
# Import abundance data: #
##########################
infile = "/work/projects/LHS/AAD/export/taxonomy/otu_table_final_rarefied_rel_L6_v2a.txt"
fhand = open(infile, 'r')
data = np.array([], dtype=np.float64)
taxa = []
i = 0
for line in fhand:
    line = line.rstrip()
    row = line.split("\t")
    
    if i == 0:
        sample_ids = row
        sample_ids.pop(0)
    elif i == 1:
        data = np.array(row[1:len(row)], dtype=float)
    else:
        taxa.append(row[0])
        data = np.vstack((data, np.array(row[1:len(row)])))


    i = i + 1
print("shape data: ")
data = np.transpose(data)
print(data.shape)
#print(sample_ids)

############################
# Import mapping file data #
############################
infile_mapping = "/work/projects/LHS/AAD/mapping_file_status2.tsv"
fhand_m = open(infile_mapping, 'r')
mapping = np.array([], dtype=object)
variable = "VisitCategory"
variable_index = int

i = 0
for line in fhand_m:
    line = line.rstrip()
    row = line.split("\t")
    
    if i == 0:
        header = row
        variable_index = header.index(variable)
        
    elif i == 1:
        mapping = np.array(row[0:len(row)], dtype=object)
    else:
        mapping2 = np.array(row[0:len(row)], dtype=object)
        mapping = np.vstack((mapping, mapping2))
        #print(mapping2)

    i = i + 1
 
# make sure we have the same samples in data, mapping and sample_ids
idxs = []
idxs2 = []

for i,v in enumerate(sample_ids):
    # Do mapping
    if str(list(mapping.T[0])).find(v) != -1:
        k = list(mapping.T[0]).index(v)
        idxs.append(k)
        idxs2.append(i)
        #sample_ids_present.append(v)
    else:
        
        print(v + " was not found!!!")

#print(idxs2)    
sample_ids_present = np.array(sample_ids)[idxs2]
X = data[idxs2]

y = mapping[:,variable_index]
print(sample_ids_present.shape)
print(X.shape)
print(y.shape)

# Before going further, we'll create a simple key equivalence scheme for classes/anwers
# here normal = 0 and altered = 1. Then we can split the data and train the model.
ys = np.unique(y)
y2 = []
for i,v in enumerate(y):
    k = list(ys).index(v)
    y2.append(k)

# make sure to force conversion to appropriate data type before fitting the model.
y2 = np.array(y2, dtype=int)
X = np.array(X, dtype=np.float64)
# sicne data contains many 0, lets add 1 to each cell.
X = X + 1
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=123)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train.dtype)
print(y_train.dtype)

nb = NaiveBayes()
# train data
nb.fit(X_train, y_train)
# And then predict outcome of test data using training model.
predictions = nb.predict(X_test)
print("Accuracy:")
print(accuracy(y_test, np.array(predictions)))