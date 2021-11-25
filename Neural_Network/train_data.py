############################
# written by Trang H. Tran and Lam M. Nguyen
############################

""" 
Training
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from Lenet import *
from schedule_LR import *

def train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path):
    # Loading data and start network
    listeta, listalpha, listbeta = params
    #-------------------------------------------------------------------------------
    print('Step 0: Load data and start network')
    # Load data
    train_loader, test_loader = load_data (dataname, 256)

    # Start network
    if (dataname == 'cifar10'):
      net = LeNet5()
      name_net = LeNet5    
    if (dataname == 'fashionmnist'):
      net = LeNet_300_100()
      name_net = LeNet_300_100
    print('---', name_net, 'started')

    # GPU :)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Loss Function
    criterion = nn.CrossEntropyLoss()

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
              SMG_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, beta, criterion, record_path, record_seed)
            elif namealg == '_SGDM_':
              SGDM_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, beta, criterion, record_path, record_seed)
            elif namealg == '_SGD_':
              SGD_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, criterion, record_path, record_seed)
            elif namealg == '_ADAM_':
              ADAM_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, criterion, record_path, record_seed)
          
    return listrecord