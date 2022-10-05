#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to train vision-based safety model using SVM
# -----------------------------------------

from __future__ import print_function
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from ignite.metrics import Precision, Recall
import time
import os
import getpass
from sklearn.svm import SVC
import pickle
import math
import sys

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
if not os.path.exists(path + 'safety_estimator/logs/SVM_log/'):
    os.mkdir(path + 'safety_estimator/logs/SVM_log/')

init_time = str(int(time.time()))
fidout = open(path + 'safety_estimator/logs/SVM_log/' + 'log_' + init_time + '.txt', 'a')

# -----------------------------------------
# load matrix for training data (classA + classB)
# -----------------------------------------
X_train = np.load(path + 'safety_estimator/X_train_new_scenario_5_6.npy')
Y_train = np.load(path + 'safety_estimator/Y_train_new_scenario_5_6.npy')

# old Y: -1 or 1 -> new Y: 0 or 1
temp_Y_train = []
for item in Y_train:
    if item == -1.0:
        temp_Y_train.append(0.0)
    else:
        temp_Y_train.append(item)
Y_train = temp_Y_train

# -----------------------------------------
# load matrix for test data (classA + classB)
# -----------------------------------------
X_test = np.load(path + 'safety_estimator/X_test_new_scenario.npy')
Y_test = np.load(path + 'safety_estimator/Y_test_new_scenario.npy')

# old Y: -1 or 1 -> new Y: 0 or 1
temp_Y_test = []
for item in Y_test:
    if item == -1.0:
        temp_Y_test.append(0.0)
    else:
        temp_Y_test.append(item)
Y_test = temp_Y_test

# -----------------------------------------
# start training
# -----------------------------------------
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)

# -----------------------------------------
# save trained model
# -----------------------------------------
# with open(path + 'models/pretrained_safety_model_town05_SVM_5_6.pkl','wb') as f:
#     pickle.dump(clf,f)
    
# # -----------------------------------------
# # load trained model
# # -----------------------------------------
# with open(path + 'models/pretrained_safety_model_town05_SVM_5_6.pkl', 'rb') as f:
#     clf = pickle.load(f)

# -----------------------------------------
# start testing
# -----------------------------------------
# collision:0 no_collison = 1
TP = 0 # true collision
TN = 0 # true no_collision
FP = 0 # false collision
FN = 0 # false no_collision

for index in range(len(X_test)):
    predicted_output = clf.predict([X_test[index]])
    true_output = Y_test[index]
    if predicted_output[0] == 0. and true_output == 0.:
        TP += 1
    elif predicted_output[0] == 0. and true_output == 1.:
        FP += 1
    elif predicted_output[0] == 1. and true_output == 0.:
        FN += 1
    elif predicted_output[0] == 1. and true_output == 1.:
        TN += 1
    else:
        print('predicted_output: {} true_output: {}'.format(predicted_output[0], true_output))
        print('Error: wrong results')
        
print('TP:{} TN:{} FP:{} FN:{}'.format(TP, TN, FP, FN))
print('accuracy of SVM: {:0.2f}%'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
print('precision of SVM: {:0.2f}%'.format(100 * (TP) / (TP + FP)))
print('recall of SVM: {:0.2f}%'.format(100 * (TP) / (TP + FN)))

fidout.write('TP:{} TN:{} FP:{} FN:{}\n'.format(TP, TN, FP, FN))
fidout.write('accuracy of SVM: {:0.2f}%\n'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
fidout.write('precision of SVM: {:0.2f}%\n'.format(100 * (TP) / (TP + FP)))
fidout.write('recall of SVM: {:0.2f}%\n'.format(100 * (TP) / (TP + FN)))