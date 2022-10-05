#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to random select N files.
# -----------------------------------------

from __future__ import print_function
import numpy as np
import getpass
import os
import sys
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import random
import shutil

# -----------------------------------------
# import customized functions
# -----------------------------------------
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils

np.random.seed(42)

# -----------------------------------------
# training settings
# -----------------------------------------
params = {
    'flag_counter_train_classA': 470, # number of items used in training classA; 100%:7673; 70%:5371; 40%:3069; 10%:767;
    'flag_counter_train_classB': 420, # number of items used in training classB; 100%:6785; 70%:4749; 40%:2714; 10%:678;
    'flag_counter_test': 99999, # number of items used in test
    'flag_test': False, # False: not collected; True: collected
    'interval': 500 # number of items for print
}
N_classA = 4700
N_classB = 4204

# -----------------------------------------
# load data
# -----------------------------------------
path = utils.root_path()

 # load training data
train_path_classA = path + 'safety_estimator/train_data_town5_1-6/classA/'
train_path_classB = path + 'safety_estimator/train_data_town5_1-6/classB/'

# -----------------------------------------
# create matrix for training data (classA)
# -----------------------------------------
counter = 0

while counter <= N_classA - params['flag_counter_train_classA']:
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(train_path_classA):
        filename_list.extend(filenames)
    filename_list.extend(filenames)
    rm_filename = random.choice(filename_list)
    print('rm_filename:{}'.format(rm_filename))
    os.remove(train_path_classA + rm_filename)
    counter += 1

# -----------------------------------------
# create matrix for training data (classB)
# -----------------------------------------
counter = 0

while counter <= N_classB - params['flag_counter_train_classB']:
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(train_path_classB):
        filename_list.extend(filenames)

    filename_list.extend(filenames)
    rm_filename = random.choice(filename_list)
    print('rm_filename:{}'.format(rm_filename))
    os.remove(train_path_classB + rm_filename)
    counter += 1

print('data generation is done!')