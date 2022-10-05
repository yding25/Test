#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to classify data into 
# - train_data_town5
#   - ClassA 
#   - ClassB
# - test_data_town5
#   - ClassA 
#   - ClassB
# -----------------------------------------

from __future__ import print_function
import os
import getpass
import shutil
import re
import sys

# -----------------------------------------
# import customized functions
# -----------------------------------------
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils

path = utils.root_path()

# [, , , , , , , , , , , ,, , ,,  , , , , , , ]:
    
foldername ='raw_data_76_75'
percent = 0.7 / 6. * 2.
# -----------------------------------------
# classify "raw_data/raw_frame" into "no_collision" and "exist_collision"
# -----------------------------------------
fidin = open(path + 'safety_estimator/' + foldername + '/overview.txt', 'r')
# create two folders
if not os.path.exists(path + 'safety_estimator/' + foldername + '/exist_collision/'):
    os.mkdir(path + 'safety_estimator/' + foldername + '/exist_collision/')
if not os.path.exists(path + 'safety_estimator/' + foldername + '/no_collision/'):
    os.mkdir(path + 'safety_estimator/' + foldername + '/no_collision/')

# absolute path
current_path = path + 'safety_estimator/' + foldername + '/raw_frame/'
target_path_no = path + 'safety_estimator/' + foldername + '/no_collision/'
target_path_exist = path + 'safety_estimator/' + foldername + '/exist_collision/'

# get filename
filename_list = []
for (dirpath, dirnames, filenames) in os.walk(current_path):
    filename_list.extend(filenames)

# record the number of two group cases
num_no = 0
num_exist = 0

# record time of two group cases
time_no = []
time_exist = []

# start classifying
for line in fidin.readlines():
    line = line.strip()
    line = line.split(':')
    if 'no' in line[0]:
        time_in_name = line[1]
        time_in_name = time_in_name[1:]
        time_no.append(time_in_name)
        for name in filename_list:
            if time_in_name in name:
                shutil.copy(current_path + name, target_path_no + name)
        num_no += 1
    else:
        time_in_name = line[1]
        time_in_name = time_in_name[1:]
        time_exist.append(time_in_name)
        for name in filename_list:
            if time_in_name in name:
                shutil.copy(current_path + name, target_path_exist + name)
        num_exist += 1
print('number of no collision: {}, number of collision: {}'.format(num_no, num_exist))

# -----------------------------------------
# classify data into "train_data_town5" and "test_data_town5"
# -----------------------------------------
# create two folders to store training and test data
if not os.path.exists(path + 'safety_estimator/train_data_town5/'):
    os.mkdir(path + 'safety_estimator/train_data_town5/')
if not os.path.exists(path + 'safety_estimator/test_data_town5/'):
    os.mkdir(path + 'safety_estimator/test_data_town5/')

# -----------------------------------------
# classify "raw_data/no_collision" into "train_data_town5/classA" and "test_data_town5/classA"
# -----------------------------------------
# create two folders
if not os.path.exists(path + 'safety_estimator/train_data_town5/classA/'):
    os.mkdir(path + 'safety_estimator/train_data_town5/classA/')
if not os.path.exists(path + 'safety_estimator/test_data_town5/classA/'):
    os.mkdir(path + 'safety_estimator/test_data_town5/classA/')

# absolute path
current_path_classA = path + 'safety_estimator/' + foldername + '/no_collision/'
target_path_train_classA = path + 'safety_estimator/train_data_town5/classA/'
target_path_test_classA = path + 'safety_estimator/test_data_town5/classA/'

# get filename
filename_list_classA = []
for (dirpath, dirnames, filenames) in os.walk(current_path_classA):
    filename_list_classA.extend(filenames)

# 70 % for train and 30 % for test
filename_no_train_classA = filename_list_classA[0:int(len(filename_list_classA)*percent)]
filename_no_test_classA = filename_list_classA[int(len(filename_list_classA)*percent):]

# start classifying
num_no_train_classA = 0
num_no_test_classA = 0
for file in filename_no_train_classA:
    shutil.copy(current_path_classA + file, target_path_train_classA + file)
    num_no_train_classA += 1
for file in filename_no_test_classA:
    shutil.copy(current_path_classA + file, target_path_test_classA + file)
    num_no_test_classA += 1
print('number of no_train_classA: {}, number of no_test_classA: {}'.format(num_no_train_classA, num_no_test_classA))

# -----------------------------------------
# classify "raw_data/exist_collision" into "train_data_town5/classB" and "test_data_town5/classB"
# -----------------------------------------
# create two folders
if not os.path.exists(path + 'safety_estimator/train_data_town5/classB/'):
    os.mkdir(path + 'safety_estimator/train_data_town5/classB/')
if not os.path.exists(path + 'safety_estimator/test_data_town5/classB/'):
    os.mkdir(path + 'safety_estimator/test_data_town5/classB/')

# absolute path
current_path_classB = path + 'safety_estimator/' + foldername + '/exist_collision/'
target_path_train_classB = path + 'safety_estimator/train_data_town5/classB/'
target_path_test_classB = path + 'safety_estimator/test_data_town5/classB/'

# get filename
filename_list_classB = []
for (dirpath, dirnames, filenames) in os.walk(current_path_classB):
    filename_list_classB.extend(filenames)

# 70 % for train and 30 % for test
filename_exist_train_classB = filename_list_classB[0:int(len(filename_list_classB)*percent)]
filename_exist_test_classB = filename_list_classB[int(len(filename_list_classB)*percent):]

# start classifying
num_exist_train_classB = 0
num_exist_test_classB = 0
for file in filename_exist_train_classB:
    shutil.copy(current_path_classB + file, target_path_train_classB + file)
    num_exist_train_classB += 1
for file in filename_exist_test_classB:
    shutil.copy(current_path_classB + file, target_path_test_classB + file)
    num_exist_test_classB += 1
print('number of exist_train_classB: {}, number of exist_test_classB: {}'.format(num_exist_train_classB, num_exist_test_classB))