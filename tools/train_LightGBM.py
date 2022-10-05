#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to train vision-based safety model using lgbmBoost
# -----------------------------------------

from __future__ import print_function
import getpass
import os
import sys
import time
from sklearn import metrics
import pickle
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot

# -----------------------------------------
# import customized functions
# -----------------------------------------
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils

np.random.seed(42)

# -----------------------------------------
# save running results
# -----------------------------------------
global path, fidout

path = utils.root_path()

if not os.path.exists(path + 'safety_estimator/logs/'):
    os.mkdir(path + 'safety_estimator/logs/')
if not os.path.exists(path + 'safety_estimator/logs/LgbmBoost_log/'):
    os.mkdir(path + 'safety_estimator/logs/LgbmBoost_log/')

init_time = str(int(time.time()))
fidout = open(path + 'safety_estimator/logs/LgbmBoost_log/' + 'log_' + init_time + '.txt', 'a')

# -----------------------------------------
# training settings
# -----------------------------------------
params = {
    'learning_rate': 0.1, # learning rate for training
    'n_estimators': 50 # number of weak learners to train iteratively
}

# -----------------------------------------
# load matrix for training data (classA + classB)
# -----------------------------------------
X_train = np.load(path + 'safety_estimator/X_train.npy')
Y_train = np.load(path + 'safety_estimator/Y_train.npy')

# old Y: -1 or 1 -> new Y: 0 or 1
temp_Y_train = []
for item in Y_train:
    if item == -1.0:
        temp_Y_train.append(0.0)
    else:
        temp_Y_train.append(item)
Y_train = temp_Y_train

# -----------------------------------------
# start training
# -----------------------------------------
lgbm = LGBMClassifier()
lgbm.fit(X_train, Y_train)

# -----------------------------------------
# save trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_lgbmBoost.pkl','wb') as f:
    pickle.dump(lgbm, f)

# -----------------------------------------
# load matrix for test data (classA + classB)
# -----------------------------------------
X_test = np.load(path + 'safety_estimator/X_test.npy')
Y_test = np.load(path + 'safety_estimator/Y_test.npy')

# old Y: -1 or 1 -> new Y: 0 or 1
temp_Y_test = []
for item in Y_test:
    if item == -1.0:
        temp_Y_test.append(0.0)
    else:
        temp_Y_test.append(item)
Y_test = temp_Y_test

# -----------------------------------------
# load trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_lgbmBoost.pkl', 'rb') as f:
    lgbm = pickle.load(f)

# -----------------------------------------
# start testing
# -----------------------------------------
# collision:0 no_collison = 1
TP = 0 # true collision
TN = 0 # true no_collision
FP = 0 # false collision
FN = 0 # false no_collision

for index in range(len(X_test)):
    predicted_output = lgbm.predict([X_test[index]])
    true_output = Y_test[index]
    
    if predicted_output[0] == 0 and true_output == 0:
        TP += 1
    elif predicted_output[0] == 0 and true_output == 1:
        FP += 1
    elif predicted_output[0] == 1 and true_output == 0:
        FN += 1
    elif predicted_output[0] == 1 and true_output == 1:
        TN += 1
    else:
        print('Error: wrong results')

print('TP:{} TN:{} FP:{} FN:{}'.format(TP, TN, FP, FN))
print('accuracy of LgbmBoost: {:0.2f}%'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
print('precision of LgbmBoost: {:0.2f}%'.format(100 * (TP) / (TP + FP)))
print('recall of LgbmBoost: {:0.2f}%'.format(100 * (TP) / (TP + FN)))

fidout.write('TP:{} TN:{} FP:{} FN:{}\n'.format(TP, TN, FP, FN))
fidout.write('accuracy of LgbmBoost: {:0.2f}%\n'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
fidout.write('precision of LgbmBoost: {:0.2f}%\n'.format(100 * (TP) / (TP + FP)))
fidout.write('recall of LgbmBoost: {:0.2f}%\n'.format(100 * (TP) / (TP + FN)))