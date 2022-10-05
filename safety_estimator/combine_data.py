#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is combine all overview.txt together
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

path = utils.root_path()

# create one folder: raw_data_full
if not os.path.exists(path + 'safety_estimator/raw_data_full/'):
    os.mkdir(path + 'safety_estimator/raw_data_full/')
folder0 = (path + 'safety_estimator/raw_data_full/')
# create overview.txt
fidout = open(folder0 + 'overview_temp.txt', 'w')
# create one folder: raw_frame
if not os.path.exists(path + 'safety_estimator/raw_data_full/raw_frame/'):
    os.mkdir(path + 'safety_estimator/raw_data_full/raw_frame/')
target_path = folder0 + 'raw_frame/'


# 2K: 'raw_data_4_3', 'raw_data_8_7', 'raw_data_17_18', 'raw_data_37_38', 'raw_data_41_42', 'raw_data_76_75'
# 0.1k: 'raw_data_72_71', 'raw_data_131_132', 'raw_data_135_136', 'raw_data_107_108', 'raw_data_111_112', 'raw_data_153_152', 'raw_data_148_147', 'raw_data_144_143', 'raw_data_126_125', 'raw_data_122_121', 'raw_data_129_130', 'raw_data_118_117', 'raw_data_106_105', 'raw_data_12_11', 'raw_data_16_15', 'raw_data_84_83', 'raw_data_36_35', 'raw_data_52_51'
for folder_name in ['raw_data_4_3', 'raw_data_8_7', 'raw_data_17_18', 'raw_data_37_38', 'raw_data_41_42', 'raw_data_72_71', 'raw_data_131_132', 'raw_data_135_136', 'raw_data_107_108', 'raw_data_111_112', 'raw_data_153_152', 'raw_data_148_147', 'raw_data_144_143', 'raw_data_126_125', 'raw_data_122_121', 'raw_data_129_130', 'raw_data_118_117', 'raw_data_106_105', 'raw_data_12_11', 'raw_data_16_15', 'raw_data_84_83', 'raw_data_36_35', 'raw_data_52_51', 'raw_data_76_75']:
    folder1 = (path + 'safety_estimator/' + folder_name + '/')
    # write to overview.txt
    fidin = open(folder1 + 'overview.txt', 'r')
    for line in fidin.readlines():
        fidout.write(line)
    # copy file to raw_frame
    filename_list = []
    current_path = folder1 + 'raw_frame/'
    for (dirpath, dirnames, filenames) in os.walk(current_path):
        filename_list.extend(filenames)
    for name in filename_list:
        shutil.copy(current_path + name, target_path + name)

# shuffle overview.txt
fidin = open(folder0 + 'overview_temp.txt', 'r')
fidout = open(folder0 + 'overview.txt', 'w')
overall = []
for line in fidin.readlines():
    overall.append(line.strip())
random.shuffle(overall)
for item in overall:
    fidout.write(item)
    fidout.write('\n')
print('data generation is done!')