#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:28:47 2023

@author: tremblayju
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=15, random_state=4)
X = np.sqrt(X**2)
X = X * 10
y = np.sqrt(y**2)
y = y / 10

plt.scatter(X[:,0],y)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(X_train)
print(y_train)
#dataset = [[0.5, 1.4],
#           [2.3, 1.9],
#           [2.9, 3.2]]

#X_train = np.array([[0.5], [2.3], [2.9]])
#X_train = np.array([[0.5], [2.3], [2.9]])
#y_train = np.array([1.4, 1.9, 3.2])
#y_test = np.array([1.4, 1.9, 3.2])

print(X_train.shape)
print(y_train.shape)

#print(X_train)

from linear_regression import LinearRegression

# Instanciate regressor object
my_lrs = [0.005, 0.004, 0.0025, 0.001]#, 0.01, 0.003, 0.001,0.0001]
my_n_iter = 25
results = {}
for my_lr in my_lrs:
    print("DEBUG: training for learning rate (alpha) = " + str(my_lr))
    regressor = LinearRegression(lr=my_lr, n_iters=my_n_iter)
    # fit model with training data
    regressor.fit(X_train, y_train)
    # test model with test data (i.e. test prediction of the model.) 
    predicted = regressor.predict(X_test)
    results[my_lr] = regressor

##########################################################
# Here we loop for the learning rate (i.e. lr or alpha)  #
# that converge in less number of iterations.            #
#                                                        #
#                                                        #
##########################################################
my_weights_per_iter = [results[res].w_res_list for res in results]
i = 0
my_weights_per_iter = {}
for lr in results:
    #my_weights_per_iter[lr] = results[lr].w_res_list
    plt.plot(list(range(my_n_iter)),  results[lr].w_res_list, label='Alpha =  %s' % lr)
    
plt.xlabel('Number of iterations')
plt.ylabel("loss (i.e. Squared residual)")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid()
plt.show()

# Here, the best model is the one with 0.5 (0.65 seems a little bit shaky near its dropping phase towards convergence)
predicted = results[0.004].predict(X_test)
# Next, compute Mean Squared Error (i.e. total error when comparing y_true vs y)
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value = mse(y_test, predicted)
print("DEBUG: mse_value -> " + str(mse_value))

# Test model with various coefficients
w = results[0.004].weight[0]
b = results[0.004].bias
print(w, b)
my_weights = np.arange(w - 0.7, w + 0.3, 0.1).tolist()
print(my_weights)
my_losses = []
for my_weight in my_weights:
    #y = mx + b
    #y = my_weight * X_test + b
    my_y_predicted = np.dot(X_test, my_weight) + b
    #my_mse = mse(y_test, my_y_predicted)
    my_loss = np.mean((y_test - my_y_predicted)**2)
    #print("DEBUG my_loss: " + str(my_loss))
    my_losses.append(my_loss)
print("DEBUG my_y_predicted: " + str(my_y_predicted))
print("DEBUG my_losses: " + str(my_losses))

plt.scatter(my_weights, my_losses, label='MSE in function of weights(m)')
plt.xlabel('Weights (slope m)')
plt.ylabel("loss (i.e. Squared residual)")

# finally plot regression lines.
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
for my_weight in my_weights:
    y_pred_line = np.dot(X, my_weight) + b
    plt.plot(X, y_pred_line,  linewidth=2, label="Prediction")
plt.show()

# show 3D plot (with both derivatives of w and b)
slope = np.arange(0.1, 2.0, 0.095)  
bias = np.arange(-7, 7, 0.7)  
w0, w1 = np.meshgrid(slope, bias)  
print("weight_list:" + str(results[0.004].weight_list))
print("bias_list:" + str(results[0.004].bias_list))
print("w_res_list:" + str(results[0.004].w_res_list))
 
J = np.zeros(w0.shape)
m = X.shape[0]
for i in range(J.shape[0]):
    for j in range(J.shape[0]):
        y_ = (w0[i,j] * X) + w1[i,j]
        J[i,j] = np.sum((y_ - y_train)**2)/m
        
fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.plot_surface(w0, w1, J, cmap="rainbow")
axes.scatter(results[0.004].weight_list, results[0.004].bias_list, results[0.004].w_res_list)
axes.set_xlabel('Slope')  
axes.set_ylabel('Bias')  
plt.show

fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.contour(w0, w1, J, cmap="rainbow")
axes.scatter(results[0.004].weight_list, results[0.004].bias_list, results[0.004].w_res_list)
axes.set_xlabel('Slope')  
axes.set_ylabel('Bias')  
plt.show

fig=plt.figure()
plt.contour(w0, w1, J, cmap="rainbow")
plt.scatter(results[0.004].weight_list, results[0.004].bias_list)
plt.show

