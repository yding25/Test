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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

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
    'learning_rate': 0.1, # learning rate for training
    'n_estimators': 50 # number of weak learners to train iteratively
}

path = utils.root_path()

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
# load trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_SVM.pkl', 'rb') as f:
    neigh = pickle.load(f)

# -----------------------------------------
# start testing
# -----------------------------------------
counter_success = 0
counter_failure = 0

# collision:0 no_collison = 1
TP = 0 # true collision
TN = 0 # true no_collision
FP = 0 # false collision
FN = 0 # false no_collision

temp_FP_value = []
temp_TP_value = []
temp_FN_value = []
temp_TN_value = []

for index in range(len(X_test)):    
    predicted_output = neigh.predict([X_test[index]])
    # print('predicted_output:{}'.format(predicted_output))
    true_output = Y_test[index]

    value = neigh.predict_proba([X_test[index]])[0]
    predict_value = predicted_output[0]
    target_value = true_output
    if (value[0] > value[1]) and (predict_value == target_value):
        temp_TP_value.append(value[0])
    elif (value[0] > value[1]) and (predict_value != target_value):
        temp_FP_value.append(value[0])
    elif (value[0] < value[1]) and (predict_value == target_value):
        temp_TN_value.append(value[1])
    elif (value[0] < value[1]) and (predict_value != target_value):
        temp_FN_value.append(value[1])
    
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

np.save(path + 'safety_estimator/distribution/FP_value_SVM.npy', temp_FP_value)
np.save(path + 'safety_estimator/distribution/TP_value_SVM.npy', temp_TP_value)
np.save(path + 'safety_estimator/distribution/FN_value_SVM.npy', temp_FN_value)
np.save(path + 'safety_estimator/distribution/TN_value_SVM.npy', temp_TN_value)


print('TP:{} TN:{} FP:{} FN:{}'.format(TP, TN, FP, FN))
print('accuracy of SVM: {:0.2f}%'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
print('precision of SVM: {:0.2f}%'.format(100 * (TP) / (TP + FP)))
print('recall of SVM: {:0.2f}%'.format(100 * (TP) / (TP + FN)))


# -----------------------------------------
# load data 0: safe 1: risk
# -----------------------------------------
path = utils.root_path()

FP_value = np.load(path + 'safety_estimator/distribution/FP_value_SVM.npy')
TP_value = np.load(path + 'safety_estimator/distribution/TP_value_SVM.npy')
FN_value = np.load(path + 'safety_estimator/distribution/FN_value_SVM.npy')
TN_value = np.load(path + 'safety_estimator/distribution/TN_value_SVM.npy')
FN_value = 1.0 - FN_value
TN_value = 1.0 - TN_value
# print('FP_value:{}'.format(FP_value))
# print('TP_value:{}'.format(TP_value))
# print('FN_value:{}'.format(FN_value))
# print('TN_value:{}'.format(TN_value))
# -----------------------------------------
# some settings
# -----------------------------------------
bar_num = 5
bar_width = 0.45
error_capsize = 5
xaxis_fontsize = 12
yaxis_fontsize = 12
legend_fontsize = 12
figure_weight = 12
figure_height = 6
xaxis_degrees = 0 # degrees of labels for X values
yaxis_degrees = 90
grid = True
ylim_min = 0.
ylim_max = 0.5
fig_format1 = 'svg'
fig_format2 = 'pdf'
fig_transparent = False
filename = 'fig_softmax'
colors = ['#cbe6b6', '#ff8243', '#c043ff', '#82ff43']

fig, ax = plt.subplots(2, 2, figsize=(figure_weight, figure_height))

# positive
FP_counts, FP_bins = np.histogram(FP_value, bar_num)
TP_counts, TP_bins = np.histogram(TP_value, bar_num)

FP_counts = FP_counts / len(FP_value)
TP_counts = TP_counts / len(TP_value)
np.save(path + 'interaction/setting/FP_probs_SVM.npy', FP_counts)
np.save(path + 'interaction/setting/FP_bins_SVM.npy', FP_bins)
np.save(path + 'interaction/setting/TP_probs_SVM.npy', TP_counts)
np.save(path + 'interaction/setting/TP_bins_SVM.npy', TP_bins)
# print('-'*30)
# print('FP_counts: {} FP_bins: {}'.format(FP_counts, FP_bins))
# print('TP_counts: {} TP_bins: {}'.format(TP_counts, TP_bins))

# negative
FN_counts, FN_bins = np.histogram(FN_value, bar_num)
TN_counts, TN_bins = np.histogram(TN_value, bar_num)

FN_counts = FN_counts / len(FN_value)
TN_counts = TN_counts / len(TN_value)
np.save(path + 'interaction/setting/FN_probs_SVM.npy', FN_counts)
np.save(path + 'interaction/setting/FN_bins_SVM.npy', FN_bins)
np.save(path + 'interaction/setting/TN_probs_SVM.npy', TN_counts)
np.save(path + 'interaction/setting/TN_bins_SVM.npy', TN_bins)
# print('-'*30)
# print('FN_counts: {} FN_bins: {}'.format(FN_counts, FN_bins))
# print('TN_counts: {} TN_bins: {}'.format(TN_counts, TN_bins))

# plot
# print('FP_value:{}'.format(FP_value))
ax[0][0].hist(FP_value, bins=FP_bins)
ax[0][0].set_title('FP')

# print('TP_value:{}'.format(TP_value))
ax[0][1].hist(TP_value, bins=TP_bins)
ax[0][1].set_title('TP')

ax[1][0].hist(FN_value, bins=FN_bins)
ax[1][0].set_title('FN')

ax[1][1].hist(TN_value, bins=TN_bins)
ax[1][1].set_title('TN')

# plt.show()