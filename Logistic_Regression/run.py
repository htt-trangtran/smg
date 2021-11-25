############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Run the experiments
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

# Experiment 1: Comparing SMG with Other Methods -------------------------------

namelr = 'const_'
num_epoch = [200, 10] # Run for 200 epochs, and measure the performance each 10 epochs

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'
listrecord = []

namealg = '_SMG_'
params = [[0.5, 0.4, 0.2, 0.1, 0.08, 0.06, 0.05], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_SGD_'
params = [[0.5, 0.4, 0.2], [0], [0]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_SGDM_'
params = [[0.05, 0.04, 0.02, 0.01, 0.008, 0.006, 0.005], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_ADAM_'
params = [[0.002, 0.001, 0.0005], [0], [0]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Data: ijcnn1 ----------------------------------------------------------------
dataname = 'ijcnn1'
listrecord = []

namealg = '_SMG_'
params = [[0.5, 0.4, 0.2, 0.1, 0.08, 0.06, 0.05], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_SGD_'
params = [[0.5, 0.4, 0.2], [0], [0]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_SGDM_'
params = [[0.05, 0.04, 0.02, 0.01, 0.008, 0.006, 0.005], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namealg = '_ADAM_'
params = [[0.002, 0.001, 0.0005], [0], [0]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Experiment 2: Comparing hyper-parameters for SMG -----------------------------

namelr = 'const_'
namealg = '_SMG_'
num_epoch = [200, 10]

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'
listrecord = []

params = [[0.2, 0.1, 0.05], [0], [0.1, 0.2, 0.3, 0.4, 0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Data: ijcnn1 ----------------------------------------------------------------
dataname = 'ijcnn1'
listrecord = []

params = [[0.2, 0.1, 0.05], [0], [0.1, 0.2, 0.3, 0.4, 0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Experiment 3: Comparing learning rate schemes for SMG ------------------------

namealg = '_SMG_'
num_epoch = [200, 10]

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a' 
listrecord = []

namelr = 'const_'
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namelr = 'cos_'
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [num_epoch[0]], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namelr = 'exp_'
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [0.99, 0.995, 0.999], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namelr = 'dim_' 
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [1, 2, 4, 8], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)
plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

# Data: ijcnn1 ----------------------------------------------------------------
dataname = 'ijcnn1' 
listrecord = []

namelr = 'const_'
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [0], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namelr = 'cos_'
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [num_epoch[0]], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namelr = 'exp_'
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [0.99, 0.995, 0.999], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)

namelr = 'dim_' 
params = [[0.5,0.4,0.2,0.1,0.08,0.06,0.05], [1, 2, 4, 8], [0.5]]
listrecord = train_data (dataname, num_epoch, namealg, namelr, params, listrecord, record_path)
plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)

