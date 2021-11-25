############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
record_history 

There are three phases to record the training process history. 
The result will be recorded in a dictionary named statistics.
phase 'initial': set up the dictionary statistics at the beginning
phase 'measure': called each time we want to record the data to the dictionary
phase 'save'   : save the dictionary when complete training. We can modify to save the weight.
"""

import pandas as pd
import numpy as np
from util_func import *


def record_history (phase, number, X, Y, X_test, Y_test, w, lamb, record_path, record_name):
    # Initialize phase:
    if (phase == 'initial'):
        global statistics
        statistics = {
        "number": [],
        "loss": [],
        "grad_loss": [],
        "acc_train": [],
        "acc_test": [] 
         }
        global num_train, num_feature, num_test
        num_train, num_feature = np.shape(X)
        num_test = len(Y_test)

    # Measure phase: 
    if (phase == 'measure'):
        # Evaluate -------------------------------------------------------------
        train_loss = func_val(num_train, num_feature, X, Y, w, lamb)
        full_grad = full_grad_eval(num_train, num_feature, X, Y, w, lamb)
        gradient_norm_square = np.dot(full_grad, full_grad)
        train_acc = accuracy(num_train, num_feature, X, Y, w, lamb)
        test_acc = accuracy(num_test, num_feature, X_test, Y_test, w, lamb)

        # Add statistics -------------------------------------------------------
        print ('--- # %d Loss %.5f Grad Loss %.5f Train & Test Acc: %.2f  %.2f' % (number, train_loss,
        gradient_norm_square, train_acc, test_acc))
        
        statistics ["number"].append(number)
        statistics ["loss"].append(train_loss)
        statistics ["grad_loss"].append(gradient_norm_square)
        statistics ["acc_train"].append(train_acc)
        statistics ["acc_test"].append(test_acc)
    
    # Save phase:
    if (phase == 'save'):
        # Save the training process history
        record_name = record_name + '.csv'
        for key, value in statistics.items():
            pd.DataFrame(value).to_csv(record_path + str(key) + '_' + record_name, index = False)