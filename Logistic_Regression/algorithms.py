############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Algorithms
"""

# Stochastic Gradient Methods: SMG, SGD, SGDM, ADAM

import numpy as np
import random
import pandas as pd
from util_func import *
from record_history import *
from schedule_LR import *

#-------------------------------------------------------------------------------
def SMG_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, beta, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Start training with learning rate', scheduleLR(1))
    num_train, num_feature = np.shape(X)
    num_test = len(Y_test)
    w = 0.5*np.ones(num_feature)
    m = np.zeros(num_feature)
    v = np.zeros(num_feature)

    record_history ('initial', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)

    #---------------------------------------------------------------------------
    print('Step 2: Training', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)
        # Reset list v
        m = v / num_train
        v = 0

        # Choosing shuffling or not:
        if shuffle:
            index_set = [i for i in range(0, num_train)]
            random.shuffle(index_set)
        else:
            index_set = np.random.randint(num_train, size=num_train)

        # Training
        for j in range(0, num_train):
            # Evaluate component gradient
            grad_i = grad_com_val(index_set[j], num_train, num_feature, X, Y, w, lamb)
            
            # Algorithm update
            v += grad_i 
            update = beta * m + (1 - beta) * grad_i
            w = w - lr * update

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')



#-------------------------------------------------------------------------------
def SGD_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Start training with learning rate', scheduleLR(1)) 
    num_train, num_feature = np.shape(X)
    num_test = len(Y_test)
    w = 0.5*np.ones(num_feature)
    record_history ('initial', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)

    #---------------------------------------------------------------------------
    print('Step 2: Training', num_epoch[0], 'epoch(s) for', record_name)
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Choosing shuffling or not:
        if shuffle:
            index_set = [i for i in range(0, num_train)]
            random.shuffle(index_set)
        else:
            index_set = np.random.randint(num_train, size=num_train)

        # Training
        for j in range(0, num_train):
            # Evaluate component gradient
            grad_i = grad_com_val(index_set[j], num_train, num_feature, X, Y, w, lamb)
            
            # Algorithm update
            w = w - lr * grad_i

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

#-------------------------------------------------------------------------------
def SGDM_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, beta, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Start training with learning rate', scheduleLR(1))
    num_train, num_feature = np.shape(X)
    num_test = len(Y_test)
    w = 0.5*np.ones(num_feature)
    m = np.zeros(num_feature)

    record_history ('initial', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)

    #---------------------------------------------------------------------------
    print('Step 2: Training', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Choosing shuffling or not:
        if shuffle:
            index_set = [i for i in range(0, num_train)]
            random.shuffle(index_set)
        else:
            index_set = np.random.randint(num_train, size=num_train)

        # Training
        for j in range(0, num_train):
            # Evaluate component gradient
            grad_i = grad_com_val(index_set[j], num_train, num_feature, X, Y, w, lamb)
            
            # Algorithm update
            m = beta * m + grad_i
            w = w - lr * m

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

#-------------------------------------------------------------------------------
def ADAM_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Start training with learning rate', scheduleLR(1))
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-8
    t = 0
    num_train, num_feature = np.shape(X)
    num_test = len(Y_test)
    w = 0.5*np.ones(num_feature)
    m = np.zeros(num_feature)
    v = np.zeros(num_feature)

    record_history ('initial', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)

    #---------------------------------------------------------------------------
    print('Step 2: Training', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Choosing shuffling or not:
        if shuffle:
            index_set = [i for i in range(0, num_train)]
            random.shuffle(index_set)
        else:
            index_set = np.random.randint(num_train, size=num_train)

        # Training
        for j in range(0, num_train):
            # Evaluate component gradient
            grad_i = grad_com_val(index_set[j], num_train, num_feature, X, Y, w, lamb)
            
            # Algorithm update
            t += 1
            m = beta_1 * m + (1 - beta_1) * grad_i
            v = beta_2 * v + (1 - beta_2) * np.multiply(grad_i,grad_i)
            m_hat = m / (1 - np.power(beta_1,t))
            v_hat = v / (1 - np.power(beta_2,t))
            w = w - lr * np.divide(m_hat, np.sqrt(v_hat) + eps)           

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

