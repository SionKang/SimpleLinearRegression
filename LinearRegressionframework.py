#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:32:04 2019

@author: sionkang
"""

from math import sqrt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def tt_split(dataset, split):
    X = dataset['YearsExperience']
    Y = dataset['Salary']
    
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = split, random_state = 101)
    
    return train_X, test_X, train_Y, test_Y

def accuracy(ypredictionList, test_Y):
    percentageDiff = 0.0
    
    for i in range(len(test_Y)):
        perctemp = (ypredictionList[i] - test_Y[i])/(test_Y[i])
        percentageDiff += np.abs(perctemp)
    
    percentageDiff = (percentageDiff/len(test_Y)) * 100
    
    return percentageDiff
        
 
def rmse_metric(ypredictionList, test_Y):
    J = 0.0
    
    for i in range(len(test_Y)):
        J += (ypredictionList[i] - test_Y[i])**2
        
    J = J/len(test_Y)
    
    J = np.sqrt(J)
    
    return J

def mean(xvalues, yvalues):

    mean_x = sum(xvalues)/len(xvalues)

    mean_y = sum(yvalues)/len(yvalues)
    
    return mean_x, mean_y
    
    
def covariance(xvalues, mean_x, yvalues, mean_y):
    numerator = 0.0
    
    for i in range(len(xvalues)):
        numerator += (yvalues[i] - mean_y) * (xvalues[i] - mean_x)
        
    return numerator

    
def variance(xvalues, mean_x):
    denominator = 0.0
    
    for i in range(len(xvalues)):
        denominator += (xvalues[i] - mean_x)**2
        
    return denominator
        
    
def coefficients(train_X, train_Y):
    mean_x, mean_y = mean(train_X, train_Y)
    
    a1 = covariance(train_X, mean_x, train_Y, mean_y)/variance(train_X, mean_x)
    
    a0 = mean_y - (a1 * mean_x)
    
    return a1, a0
    
def simple_linear_regression(train_X, train_Y, test_X, test_Y):
    a1, a0 = coefficients(train_X, train_Y)
    
    ypredictionList = []
    
    for i in range(len(test_X)):
        ytemp = a0 + (a1 * test_X[i])
        ypredictionList.append(np.asscalar(ytemp))
        
    return ypredictionList

def evaluateData(dataset, split):
    train_X, test_X, train_Y, test_Y = tt_split(dataset, split)
    
    train_X = train_X.values
    train_X = train_X.reshape(-1,1)
    
    test_X = test_X.values
    test_X = test_X.reshape(-1,1)
    
    train_Y = train_Y.values
    train_Y = train_Y.reshape(-1,1)
    
    test_Y = test_Y.values
    test_Y = test_Y.reshape(-1,1)
    
    ypredictionList = simple_linear_regression(train_X, train_Y, test_X, test_Y)
    
    plt.scatter(test_Y,ypredictionList)
    
    J = rmse_metric(ypredictionList, test_Y)
    
    accurate = accuracy(ypredictionList, test_Y)
    
    return J, 100 - accurate
        
    
#TESTCLASS
dataset = pd.read_csv("Salary_data.csv")
split = 0.3

J, accurate = evaluateData(dataset, split)

print()
print("RMSE Value: " + str(round(np.asscalar(J),2)))
print("Accuracy: " + str(round(np.asscalar(accurate),2)) + "%")





    
