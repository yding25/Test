#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to train vision-based safety model using Logistic Regression (LR)
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
import sys
import getpass
from sklearn.svm import SVC
import pickle
import math
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import random

# -----------------------------------------
# import customized functions
# -----------------------------------------
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils

# -----------------------------------------
# save running results
# -----------------------------------------
global path, fidout

path = utils.root_path()

if not os.path.exists(path + 'safety_estimator/logs/'):
    os.mkdir(path + 'safety_estimator/logs/')
if not os.path.exists(path + 'safety_estimator/logs/LR_log/'):
    os.mkdir(path + 'safety_estimator/logs/LR_log/')

init_time = str(int(time.time()))
fidout = open(path + 'safety_estimator/logs/LR_log/' + 'log_' + init_time + '.txt', 'a')

# -----------------------------------------
# load matrix for training data (classA + classB)
# -----------------------------------------
X_train = np.load(path + 'safety_estimator/X_train.npy')
Y_train = np.load(path + 'safety_estimator/Y_train.npy')

# -----------------------------------------
# start training
# -----------------------------------------
clf = LogisticRegression(random_state=random.randint(0, 20), solver='sag').fit(X_train, Y_train)

# -----------------------------------------
# save trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_LR.pkl','wb') as f:
    pickle.dump(clf,f)

# -----------------------------------------
# load matrix for test data (classA + classB)
# -----------------------------------------
X_test = np.load(path + 'safety_estimator/X_test.npy')
Y_test = np.load(path + 'safety_estimator/Y_test.npy')

# -----------------------------------------
# load trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_LR.pkl', 'rb') as f:
    clf = pickle.load(f)

# -----------------------------------------
# start testing
# -----------------------------------------
counter_success = 0
counter_failure = 0
for index in range(len(X_test)):
    predicted_output = clf.predict([X_test[index]])
    true_output = Y_test[index]
    # print('predicted output: {}'.format(predicted_output[0]))
    # print('true_output: {}'.format(true_output))
    if predicted_output[0] == true_output:
        counter_success += 1
    else:
        counter_failure += 1
print('success rate of LR: {:0.2f}%'.format(100 * counter_success / (counter_success + counter_failure)))
fidout.write('success rate of LR: {:0.2f}%'.format(100 * counter_success / (counter_success + counter_failure)))