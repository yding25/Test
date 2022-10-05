#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to train vision-based safety model using XGBoost
# -----------------------------------------

from __future__ import print_function

import getpass
import os
import sys
import time
from sklearn import metrics
import pickle
import numpy as np
from numpy import asarray
from numpy import mean
from numpy import std
from sklearn import datasets
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------
# import customized functions
# -----------------------------------------
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils

np.random.seed(42)

# -----------------------------------------
# save running results
# -----------------------------------------
path = utils.root_path()

if not os.path.exists(path + 'safety_estimator/logs/'):
    os.mkdir(path + 'safety_estimator/logs/')
if not os.path.exists(path + 'safety_estimator/logs/XGBoost_log/'):
    os.mkdir(path + 'safety_estimator/logs/XGBoost_log/')

init_time = str(int(time.time()))
fidout = open(path + 'safety_estimator/logs/XGBoost_log/' + 'log_' + init_time + '.txt', 'a')

# -----------------------------------------
# training settings
# -----------------------------------------
params = {
    'learning_rate': 0.005, # learning rate for training
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
# start training
# -----------------------------------------
xgb = XGBClassifier(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
eval_set = [(X_train, Y_train), (X_test, Y_test)]
# xgb.fit(X_train, Y_train)
xgb.fit(X_train, Y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=eval_set, verbose=True)
# xgb.fit(X_train, Y_train, eval_metric="logloss", eval_set=eval_set, verbose=True)

# -----------------------------------------
# make predictions for test data
# -----------------------------------------
predicted_output = xgb.predict(X_test)
predicted_proba_output = xgb.predict_proba(X_test)
predictions = [round(value) for value in predicted_output]

# -----------------------------------------
# evaluate predictions
# -----------------------------------------
print('-'*30)
accuracy = accuracy_score(Y_test, predictions)
print("accuracy: {}".format(accuracy * 100.0))
precision = precision_score(Y_test, predictions)
print("precision: {}".format(accuracy * 100.0))
recall = recall_score(Y_test, predictions)
print("recall: {}".format(recall * 100.0))
f1 = f1_score(Y_test, predictions)
print("f1: {}".format(f1 * 100.0))
print('-'*30)

# -----------------------------------------
# evaluate predictions
# -----------------------------------------
fidout = open('FN_cases.txt', 'w')
counter = 0
test_filenames = np.load('filenames.npy')
for i in range(len(predictions)):
    if Y_test[i] == 0.0 and predictions[i] == 1:
        counter += 1
        print('real is collision, but predict is no collision')
        print('filename:{}'.format(test_filenames[i]))
        print('item in Y_test:{} in predictions:{} in predicted_proba_output:{}'.format(Y_test[i], predictions[i], predicted_proba_output[i]))
        print('-'*30)
        fidout.write('filename:{}\n'.format(test_filenames[i]))
        fidout.write('item in Y_test:{} in predictions:{} in predicted_proba_output:{}\n'.format(Y_test[i], predictions[i], predicted_proba_output[i]))
        fidout.write('-'*30)
        fidout.write('\n')

print('counter (%):{}'.format(counter/len(predictions)))

# -----------------------------------------
# plot
# -----------------------------------------
results = xgb.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBClassifier Log Loss')
pyplot.show()

# -----------------------------------------
# save trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_XGBoost.pkl','wb') as f:
    pickle.dump(xgb, f)

# -----------------------------------------
# load trained model
# -----------------------------------------
with open(path + 'models/pretrained_safety_model_town05_XGBoost.pkl', 'rb') as f:
    xgb = pickle.load(f)

# -----------------------------------------
# start testing
# -----------------------------------------
# collision:0 no_collison = 1
TP = 0 # true collision
TN = 0 # true no_collision
FP = 0 # false collision
FN = 0 # false no_collision

for index in range(len(X_test)):
    predicted_output = xgb.predict([X_test[index]])
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
print('accuracy of XGBoost: {:0.2f}%'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
print('precision of XGBoost: {:0.2f}%'.format(100 * (TP) / (TP + FP)))
print('recall of XGBoost: {:0.2f}%'.format(100 * (TP) / (TP + FN)))

fidout.write('TP:{} TN:{} FP:{} FN:{}\n'.format(TP, TN, FP, FN))
fidout.write('accuracy of XGBoost: {:0.2f}%\n'.format(100 * (TP + TN) / (TP + TN + FP + FN)))
fidout.write('precision of XGBoost: {:0.2f}%\n'.format(100 * (TP) / (TP + FP)))
fidout.write('recall of XGBoost: {:0.2f}%\n'.format(100 * (TP) / (TP + FN)))