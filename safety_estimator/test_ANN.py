#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to test vision-based safety model using ANN
# -----------------------------------------

from __future__ import print_function
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
import sys

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024*4)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024*4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

class SafetyEstimator:
    def __init__(self, state_dict_path="safe_estimation.pt") -> None:
        self.device = torch.device("cuda")
        self.sftmx = nn.Softmax(dim=1)
        self.estimate_model = Net()
        self.estimate_model.load_state_dict(torch.load(state_dict_path))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def predict(self, sample):
        new_arr = np.expand_dims(sample, axis=0)
        new_sample = torch.from_numpy(new_arr)
        # print(new_sample.size())
        new_sample.to(self.device)
        # label=self.estimate_model(new_sample).argmax(dim=1, keepdim=True)
        value = self.sftmx(self.estimate_model(new_sample))
        return float(value[0][1])

# safety_estimator = SafetyEstimator(state_dict_path="/home/yan/CARLA_0.9.10.1/PythonAPI/Test/models/1662165127pretrained_safety_model_town05_ANN.pt")
safety_estimator = SafetyEstimator(state_dict_path="/home/yan/CARLA_0.9.10.1/PythonAPI/Test/models/1662174683pretrained_safety_model_town05_ANN.pt")

current_path_classA = '/home/yan/CARLA_0.9.10.1/PythonAPI/Test/safety_estimator/test_data_town5/classA/'
current_path_classB = '/home/yan/CARLA_0.9.10.1/PythonAPI/Test/safety_estimator/test_data_town5/classB/'
filename_list_classA= []
for (dirpath, dirnames, filenames) in os.walk(current_path_classA):
        filename_list_classA.extend(filenames)
filename_list_classB= []
for (dirpath, dirnames, filenames) in os.walk(current_path_classB):
        filename_list_classB.extend(filenames)


counter_FP = 0
counter_TP = 0
border = 0.4
for filename in filename_list_classA:
    frames_data = np.load(current_path_classA + filename)
    p_risk = safety_estimator.predict(frames_data)
    if p_risk > border:
        counter_FP += 1
    else:
        counter_TP += 1


counter_FN = 0
counter_TN = 0
for filename in filename_list_classB:
    frames_data = np.load(current_path_classB + filename)
    p_risk = safety_estimator.predict(frames_data)
    if p_risk < border:
        counter_FN += 1
    else:
        counter_TN += 1

print('counter_FP:{}'.format(counter_FP))
print('counter_TP:{}'.format(counter_TP))
print('counter_FN:{}'.format(counter_FN))
print('counter_TN:{}'.format(counter_TN))

print('accuracy:{}'.format(1 - (counter_FN + counter_FP) / (len(filename_list_classA)+len(filename_list_classB))))
print('recall:{}'.format(counter_TP / (counter_TP + counter_FN)))
recall = counter_TP / (counter_TP + counter_FN)
print('precision:{}'.format(counter_TP / (counter_TP + counter_FP)))
precision = counter_TP / (counter_TP + counter_FP)
beta = 1.0
print('f-beta-score:{}'.format(((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))
beta = 1.1
print('f-beta-score:{}'.format(((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))
beta = 1.2
print('f-beta-score:{}'.format(((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))
beta = 1.3
print('f-beta-score:{}'.format(((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))
beta = 1.4
print('f-beta-score:{}'.format(((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))
beta = 1.5
print('f-beta-score:{}'.format(((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))