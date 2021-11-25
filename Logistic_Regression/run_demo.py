############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Run the experiments - DEMO
"""

import os
import numpy as np
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from util_func import *
from schedule_LR import *
from train_data import *
from average_and_plot import *

# Change the record path 
record_path = './SMG_record/'
record_avg_path = record_path + 'Avg/'

if not os.path.exists(record_path):
    os.makedirs(record_path)

if not os.path.exists(record_avg_path):
    os.makedirs(record_avg_path)

# Experiment DEMO: Comparing SMG with Other Methods
listrecord = []
namelr = 'const_'
num_epoch = [3, 1] # Run for only 3 epochs, and measure the performance after 1 epoch

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'

namealg = '_SMG_'
params = [[0.1], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_SGD_'
params = [[0.1], [0], [0]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_SGDM_'
params = [[0.005], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_ADAM_'
params = [[0.001], [0], [0]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path) 