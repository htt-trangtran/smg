############################
# written by Lam M. Nguyen and Trang H. Tran
############################

"""
Load Data files
"""

import numpy as np
import sys, os
import pandas as pd
from csv import reader

#-------------------------------------------------------------------------------
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

#-------------------------------------------------------------------------------
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
	

#-------------------------------------------------------------------------------
# Import data
def import_data(data_option):
    source_path = './data/' + data_option

    # Train Data
    train_x_data = load_csv(source_path + '_x_train.csv')
    for i in range(len(train_x_data[0])):
        str_column_to_float(train_x_data, i)
    X_train = np.array(train_x_data)	

    train_y_data = load_csv(source_path + '_y_train.csv')
    for i in range(len(train_y_data[0])):
        str_column_to_float(train_y_data, i)
    Y_train = np.array(train_y_data)	

    # Test Data
    test_x_data = load_csv(source_path + '_x_test.csv')
    for i in range(len(test_x_data[0])):
        str_column_to_float(test_x_data, i)
    X_test = np.array(test_x_data)	

    test_y_data = load_csv(source_path + '_y_test.csv')
    for i in range(len(test_y_data[0])):
        str_column_to_float(test_y_data, i)
    Y_test = np.array(test_y_data)	

    # Normalize data
    max_val = np.max(X_train)
    min_val = np.min(X_train)
    X_train = (X_train - min_val)/(max_val - min_val)
    X_test = (X_test - min_val)/(max_val - min_val)

    return X_train, Y_train, X_test, Y_test