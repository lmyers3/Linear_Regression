# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 09:51:29 2021

@author: Lindsey
"""

import numpy as np
import pandas as pd
import matplotlib as plt

def variable_matrix(data):
    #create bias
    bias = np.ones(shape=(num_rows, 1))
    
    #turn the featurn dataframe into a 2Darray
    data = np.array(data)
    
    #join bias and data to create X matrix
    data = np.concatenate([bias, data], axis=1)
    return data



#HANDLE TRAINING
file_name = input("Enter your training file name: ")
inFile = open(file_name, 'r')

#get number of rows and features, not including column labels and target
first_line = inFile.readline().split(' ')
num_rows = int(first_line[0])
num_features = int(first_line[1])

#example_col = num_features

#get list of all features
features = inFile.readline().split(',')[:num_features]
inFile.close()

#create a dataframe of all features, dont read first row or last column
data = pd.read_csv(file_name, usecols=features, skiprows=1)

#set dataframe to 2d matrix with bias
data = variable_matrix(data)

#set target column vector
targets =  pd.read_csv(file_name, usecols=[num_features], skiprows=1)
targets = np.array([targets][0])

#calculate weights
A = np.linalg.pinv(np.dot(data.T, data))
B = np.dot(data.T, targets)
w = np.dot(A,B)

#calculate J
A = np.dot(data, w) - targets
J = (1/num_rows) * np.dot(A.T, A)

print('Weights: ')
print(w)
print('J:')
print(J)



#HANDLE TEST
file_name = input("Enter the file name of your test file: ")
inFile = open(file_name, 'r')

#get number of rows and features, not including column labels and target
first_line = inFile.readline().split(' ')
num_rows = int(first_line[0])
num_features = int(first_line[1])

#get list of all features
features = inFile.readline().split(',')[:num_features]
inFile.close()

#create a dataframe of all features, dont read first row or last column
data = pd.read_csv(file_name, usecols=features, skiprows=1)
data = variable_matrix(data)

#make predictions for price using weights
predicted_y = np.dot(data, w)

#get actual values of y
actual_y =  pd.read_csv(file_name, usecols=[num_features], skiprows=1)
actual_y = np.array([actual_y][0])

#calculate J
A = predicted_y - actual_y
J = (1/num_rows) * np.dot(A.T, A)
print(f"J for test file: {J}")

#comparison table
comp_table = np.concatenate([predicted_y, actual_y], axis=1)
comp_table = pd.DataFrame(data=comp_table, columns=['Predicted', 'Actual'])
print(comp_table.head(25))




