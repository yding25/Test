#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ??
# 加载需要的包和环境
from __future__ import print_function
import argparse
import glob
import logging
import os
import sys
import numpy as np
import time
import math
import random
import statistics
import getpass
import weakref
import cv2
from numpy import random
from queue import Queue
from queue import Empty
import torch
from collections import deque
import torch.nn as nn
# from models.model_se import Net
from models.model_se_test import Net
from torchvision import datasets, transforms


class SafetyEstimator:
    def __init__(self, state_dict_path="/home/yan/CARLA_0.9.10.1/PythonAPI/Test/models/safe_estimation_town5_new.pt") -> None:
        self.device = torch.device("cuda")
        self.sftmx = nn.Softmax(dim=1)
        self.estimate_model = Net()
        self.estimate_model.load_state_dict(torch.load(state_dict_path))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def predict(self, sample):
        new_arr = np.expand_dims(sample, axis=0)
        new_sample = torch.from_numpy(new_arr)
        print(new_sample.size())
        new_sample.to(self.device)
        # label=self.estimate_model(new_sample).argmax(dim=1, keepdim=True)
        value = self.sftmx(self.estimate_model(new_sample))
        print('value:', value)
        return float(value[0][1])


# safety_estimator = SafetyEstimator(state_dict_path="/home/yan/CARLA_0.9.10.1/PythonAPI/Test/models"
#                                                    "/safe_estimation_town5.pt")
safety_estimator = SafetyEstimator(state_dict_path="/home/yan/CARLA_0.9.10.1/PythonAPI/Test/models"
                                                   "/safe_estimation_town5_new.pt")
frames_data1 = np.load('/home/yan/CARLA_0.9.10.1/PythonAPI/Test/data_for_model/train_data_town5/classA/1643258768_WorldSnapshot(frame=37086).npy')
frames_data2 = np.load('/home/yan/CARLA_0.9.10.1/PythonAPI/Test/data_for_model/train_data_town5/classB/1643258995_WorldSnapshot(frame=44082).npy')
p_risk1 = safety_estimator.predict(frames_data1)
p_risk2 = safety_estimator.predict(frames_data2)
print(p_risk1)
print(p_risk2)