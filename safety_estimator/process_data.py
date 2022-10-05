#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to generate X (input) and Y (output)
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
    'flag_counter_train_classA': 9999, # number of items used in training classA; 100%:7673; 70%:5371; 40%:3069; 10%:767;
    'flag_counter_train_classB': 9999, # number of items used in training classB; 100%:6785; 70%:4749; 40%:2714; 10%:678;
    'flag_counter_test': 99999, # number of items used in test
    'flag_train': True, # False: not collected; True: collected
    'flag_test': False, # False: not collected; True: collected
    'interval': 500 # number of items for print
}

# -----------------------------------------
# load data
# -----------------------------------------
path = utils.root_path()

 # load training data
train_path_classA = path + 'safety_estimator/train_data_town5_new_scenario_5_6/classA/'
train_path_classB = path + 'safety_estimator/train_data_town5_new_scenario_5_6/classB/'

# load test data
test_path_classA = path + 'safety_estimator/test_data_town5_new_scenario/classA/'
test_path_classB = path + 'safety_estimator/test_data_town5_new_scenario/classB/'

if params['flag_train']:
    # -----------------------------------------
    # create matrix for training data (classA)
    # -----------------------------------------
    counter = 0
    X_train_A = np.array([np.zeros(32768)])
    Y_train_A = np.array([])
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(train_path_classA):
        filename_list.extend(filenames)

    for file in filename_list:
        if counter <= params['flag_counter_train_classA']:
            temp = np.array([np.load(train_path_classA + file)])
            X_train_A = np.append(X_train_A, temp, axis=0)
            Y_train_A = np.append(Y_train_A, 1)
            counter += 1
            if counter % params['interval'] == 0:
                print('progress rate of reading training data (classA): {:.3f}%'.format(100 * counter / min(params['flag_counter_train_classA'], len(filename_list))))
        else:
            break
    X_train_A = np.delete(X_train_A, 0, axis=0)
    # print('X_train_A:{}, len:{}'.format(X_train_A, len(X_train_A)))
    # print('Y_train_A:{}, len:{}'.format(Y_train_A, len(Y_train_A)))

    # -----------------------------------------
    # create matrix for training data (classB)
    # -----------------------------------------

    counter = 0
    X_train_B = np.array([np.zeros(32768)])
    Y_train_B = np.array([])
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(train_path_classB):
        filename_list.extend(filenames)

    for file in filename_list:
        if counter <= params['flag_counter_train_classB']:
            temp = np.array([np.load(train_path_classB + file)])
            X_train_B = np.append(X_train_B, temp, axis=0)
            Y_train_B = np.append(Y_train_B, -1)
            counter += 1
            if counter % params['interval'] == 0:
                print('progress rate of reading training data (classB): {:.3f}%'.format(100 * counter / min(params['flag_counter_train_classB'], len(filename_list))))
        else:
            break
    X_train_B = np.delete(X_train_B, 0, axis=0)
    # print('X_train_B:{}, len:{}'.format(X_train_B, len(X_train_B)))
    # print('Y_train_B:{}, len:{}'.format(Y_train_B, len(Y_train_B)))

    # -----------------------------------------
    # create matrix for training data (classA + classB)
    # -----------------------------------------
    X_train = np.append(X_train_A, X_train_B, axis=0)
    Y_train = np.concatenate((Y_train_A, Y_train_B))
    # print('X_train:{}, len:{}'.format(X_train, len(X_train)))
    # print('Y_train:{}, len:{}'.format(Y_train, len(Y_train)))

    # -----------------------------------------
    # save matrix for training data (classA + classB)
    # -----------------------------------------
    np.save(path + 'safety_estimator/X_train_new_scenario_5_6.npy', X_train)
    np.save(path + 'safety_estimator/Y_train_new_scenario_5_6.npy', Y_train)


test_filename = []

if params['flag_test']:
    # -----------------------------------------
    # create matrix for test data (classA)
    # -----------------------------------------
    counter = 0
    X_test_A = np.array([np.zeros(32768)])
    Y_test_A = np.array([])
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(test_path_classA):
        filename_list.extend(filenames)

    for file in filename_list:
        if counter <= params['flag_counter_test']:
            temp = np.array([np.load(test_path_classA + file)])
            X_test_A = np.append(X_test_A, temp, axis=0)
            test_filename.append([file, 1])
            Y_test_A = np.append(Y_test_A, 1)
            counter += 1
            if counter % params['interval'] == 0:
                print('progress rate of reading testing data (classA): {:.3f}%'.format(100 * counter / min(params['flag_counter_test'], len(filename_list))))
        else:
            break
    X_test_A = np.delete(X_test_A, 0, axis=0)
    # print('X_test_A:{}, len:{}'.format(X_test_A, len(X_test_A)))
    # print('Y_test_A:{}, len:{}'.format(Y_test_A, len(Y_test_A)))

    # -----------------------------------------
    # create matrix for test data (classB)
    # -----------------------------------------
    counter = 0
    X_test_B = np.array([np.zeros(32768)])
    Y_test_B = np.array([])
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(test_path_classB):
        filename_list.extend(filenames)

    for file in filename_list:
        if counter <= params['flag_counter_test']:
            temp = np.array([np.load(test_path_classB + file)])
            X_test_B = np.append(X_test_B, temp, axis=0)
            test_filename.append([file, -1])
            Y_test_B = np.append(Y_test_B, -1)
            counter += 1
            if counter % params['interval'] == 0:
                print('progress rate of reading testing data (classB): {:.3f}%'.format(100 * counter / min(params['flag_counter_test'], len(filename_list))))
        else:
            break
    X_test_B = np.delete(X_test_B, 0, axis=0)
    # print('X_test_B:{}, len:{}'.format(X_test_B, len(X_test_B)))
    # print('Y_test_B:{}, len:{}'.format(Y_test_B, len(Y_test_B)))

    # -----------------------------------------
    # create matrix for test data (classA + classB)
    # -----------------------------------------
    X_test = np.append(X_test_A, X_test_B, axis=0)
    Y_test = np.concatenate((Y_test_A, Y_test_B))
    # print('X_test:{}, len:{}'.format(X_test, len(X_test)))
    # print('Y_test:{}, len:{}'.format(Y_test, len(Y_test)))

    # -----------------------------------------
    # save matrix for test data (classA + classB)
    # -----------------------------------------
    np.save(path + 'safety_estimator/X_test_no_new_scenario.npy', X_test)
    np.save(path + 'safety_estimator/Y_test_no_new_scenario.npy', Y_test)

    np.save(path + 'safety_estimator/filenames_no_new_scenario.npy', test_filename)

print('data generation is done!')