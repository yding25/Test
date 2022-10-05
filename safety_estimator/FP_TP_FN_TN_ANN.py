#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to train vision-based safety model using ANN
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

# -----------------------------------------
# import customized functions
# -----------------------------------------
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils

# -----------------------------------------
# model structure
# -----------------------------------------
class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024*4)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024*4, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024*4)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024*4, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024*4)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024*4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 64)
        self.dropout3 = nn.Dropout(0.3)
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
        x = self.dropout3(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024*4)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024*4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 8)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(8, 2)

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
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output

def test(model, device, test_loader, epoch):
    model.eval()
    running_loss = 0
    test_loss = 0
    correct = 0
    precision = Precision()
    recall = Recall()

    temp_FP_value = []
    temp_TP_value = []
    temp_FN_value = []
    temp_TN_value = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # start testing
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.nll_loss(output, target, reduction='sum')
            
            # get predicted result
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # analysis
            precision.update((pred, target))
            recall.update((pred, target))
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += test_loss.item()

            value = F.softmax(output, dim=1)
            value_list = value[:, 0:2].tolist()
            pred_list = pred[:, 0].tolist()
            pred_list = target.tolist()

            print('output:{}'.format(output))
            print('value_list:{}'.format(value_list))
            print('pred_list:{}'.format(pred_list))
            print('target:{}'.format(target))

            for index in range(len(value_list)):
                value = value_list[index]
                predict_value = pred_list[index]
                target_value = target[index]
                if (value[0] > value[1]) and (predict_value == target_value):
                    temp_TP_value.append(value[0])
                elif (value[0] > value[1]) and (predict_value != target_value):
                    temp_FP_value.append(value[0])
                elif (value[0] < value[1]) and (predict_value == target_value):
                    temp_TN_value.append(value[1])
                elif (value[0] < value[1]) and (predict_value != target_value):
                    temp_FN_value.append(value[1])
    return temp_FP_value, temp_TP_value, temp_FN_value, temp_TN_value


def sample_loader(path):
    arr = np.load(path)
    new_arr = np.expand_dims(arr, axis=0)
    return new_arr


def main():
    # training settings
    params = {
        'train_batch': 16, # input batch size for training (default: 16)
        'test_batch': 16, # input batch size for testing (default: 16)
        'epochs': 100, # number of epochs to train (default: 100)
        'learning_rate': 0.05, # learning rate (default: 0.05)
        'gamma': 0.1, # learning rate step gamma (default: 0.1)
        'step_size': 30, # period of learning rate decay (default: 30)
        'model': 0, # 1: 4096 * 64; 2: 4096 * 128; 3: 4096 * 256
        'interval': 100 # interval print one time
    }

    device = torch.device("cuda")
    transform = transforms.Compose([transforms.ToTensor()])
    
    path = utils.root_path()
    
    # load test data
    test_data = datasets.DatasetFolder(path + 'safety_estimator/test_data_town5_no_new_scenario', transform=transform, extensions=".npy", loader=sample_loader)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['test_batch'], shuffle=True)

     # load model parameter
    if params['model'] == 1:
        model = Net1().to(device)
    elif params['model'] == 2:
        model = Net2().to(device)
    elif params['model'] == 3:
        model = Net3().to(device)
    elif params['model'] == 4:
        model = Net4().to(device)
    elif params['model'] == 5:
        model = Net5().to(device)
    elif params['model'] == 0:
        model = Net0().to(device)
    else:
        print('error: Unknown model')
    print('model structure: {}'.format(model))

    model.load_state_dict(torch.load(path + 'models/1662424707pretrained_safety_model_town05_ANN.pt'))

    FP = []
    TP = []
    FN = []
    TN = []
    for epoch in range(1, params['epochs'] + 1):
        temp_FP_value, temp_TP_value, temp_FN_value, temp_TN_value = test(model, device, test_loader, epoch)
        FP = FP + temp_FP_value
        TP = TP + temp_TP_value
        FN = FN + temp_FN_value
        TN = TN + temp_TN_value
    
    # np.save(path + 'safety_estimator/distribution/FP_value_ANN.npy', FP)
    # np.save(path + 'safety_estimator/distribution/TP_value_ANN.npy', TP)
    # np.save(path + 'safety_estimator/distribution/FN_value_ANN.npy', FN)
    # np.save(path + 'safety_estimator/distribution/TN_value_ANN.npy', TN)

    # -----------------------------------------
    # load data 0: safe 1: risk
    # -----------------------------------------
    path = utils.root_path()

    FP_value = np.load(path + 'safety_estimator/distribution/FP_value_ANN.npy')
    TP_value = np.load(path + 'safety_estimator/distribution/TP_value_ANN.npy')
    FN_value = np.load(path + 'safety_estimator/distribution/FN_value_ANN.npy')
    TN_value = np.load(path + 'safety_estimator/distribution/TN_value_ANN.npy')
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
    np.save(path + 'interaction/setting/FP_probs_ANN.npy', FP_counts)
    np.save(path + 'interaction/setting/FP_bins_ANN.npy', FP_bins)
    np.save(path + 'interaction/setting/TP_probs_ANN.npy', TP_counts)
    np.save(path + 'interaction/setting/TP_bins_ANN.npy', TP_bins)
    # print('-'*30)
    # print('FP_counts: {} FP_bins: {}'.format(FP_counts, FP_bins))
    # print('TP_counts: {} TP_bins: {}'.format(TP_counts, TP_bins))

    # negative
    FN_counts, FN_bins = np.histogram(FN_value, bar_num)
    TN_counts, TN_bins = np.histogram(TN_value, bar_num)

    FN_counts = FN_counts / len(FN_value)
    TN_counts = TN_counts / len(TN_value)
    np.save(path + 'interaction/setting/FN_probs_ANN.npy', FN_counts)
    np.save(path + 'interaction/setting/FN_bins_ANN.npy', FN_bins)
    np.save(path + 'interaction/setting/TN_probs_ANN.npy', TN_counts)
    np.save(path + 'interaction/setting/TN_bins_ANN.npy', TN_bins)
    # print('-'*30)
    # print('FN_counts: {} FN_bins: {}'.format(FN_counts, FN_bins))
    # print('TN_counts: {} TN_bins: {}'.format(TN_counts, TN_bins))

    # plot
    print('FP_value:{}'.format(FP_value))
    ax[0][0].hist(FP_value, bins=FP_bins)
    ax[0][0].set_title('FP')

    print('TP_value:{}'.format(TP_value))
    ax[0][1].hist(TP_value, bins=TP_bins)
    ax[0][1].set_title('TP')

    ax[1][0].hist(FN_value, bins=FN_bins)
    ax[1][0].set_title('FN')

    ax[1][1].hist(TN_value, bins=TN_bins)
    ax[1][1].set_title('TN')

    plt.show()

if __name__ == '__main__':
    main()