############################
# written by Trang H. Tran and Lam M. Nguyen
############################

""" 
Training
"""

import numpy as np
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from util_func import *
from schedule_LR import *

def train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path):
    # Loading data and start network
    listeta, listalpha, listbeta = params
    #-------------------------------------------------------------------------------
    print('Step 0: Import data')
    # Load data
    X, Y, X_test, Y_test = import_data(dataname)
    lamb =  0.01
    shuffle = 1

    #-------------------------------------------------------------------------------
    # Start training

    for eta in listeta:
      for alpha in listalpha:
        for beta in listbeta:
          # Record name: 
          if namealg == '_SMG_' or namealg == '_SGDM_':
            alg = namealg + str(beta) +'_' + namelr
          elif namealg == '_SGD_' or namealg == '_ADAM_':
            alg = namealg + namelr

          # Pick LR scheme: 
          if namelr == 'const_':
            scheduleLR = constant (eta)
            record_name = dataname + alg + str(eta) 
          elif namelr == 'cos_':
            scheduleLR = cosine (eta, alpha)
            record_name = dataname + alg + str(eta) + '_' + str(alpha) 
          elif namelr == 'exp_':
            scheduleLR = exponential (eta, alpha)
            record_name = dataname + alg + str(eta) + '_' + str(alpha) 
          elif namelr == 'dim_':
            scheduleLR = diminishing (eta, alpha)
            record_name = dataname + alg + str(eta) + '_' + str(alpha) 

          listrecord.append(record_name)
          for seed in range(10):
            record_seed = record_name + '_seed_' + str(seed)
            # Pick algorithm 
            if namealg == '_SMG_':
              SMG_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, beta, record_path, record_seed)
            elif namealg == '_SGDM_':
              SGDM_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, beta, record_path, record_seed)
            elif namealg == '_SGD_':
              SGD_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
            elif namealg == '_ADAM_':
              ADAM_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
          
    return listrecord