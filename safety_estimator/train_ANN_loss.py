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
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(8192*4, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x=torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.sigmoid(x)
        return output


# -----------------------------------------
# save running results
# -----------------------------------------
global path, fidout

path = utils.root_path()

if not os.path.exists(path + 'safety_estimator/logs/'):
    os.mkdir(path + 'safety_estimator/logs/')
if not os.path.exists(path + 'safety_estimator/logs/ANN_log/'):
    os.mkdir(path + 'safety_estimator/logs/ANN_log/')

init_time = str(int(time.time()))
fidout = open(path + 'safety_estimator/logs/ANN_log/' + 'log_' + init_time + '.txt', 'a')


def train(params, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    correct = 0
    precision = Precision()
    recall = Recall()
    for batch_idx, (data, target) in enumerate(train_loader):
        # start training
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.float()
        target = target.unsqueeze(1)
        loss = F.binary_cross_entropy(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()

        # get predicted result
        pred = output.round()

        # analysis
        precision.update((pred, target))
        recall.update((pred, target))
        correct += pred.eq(target.view_as(pred)).sum().item()
        running_loss += loss.item()
        
        if batch_idx % params['interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx/len(train_loader)))
    
    print('-'*50)
    print('Train: loss_avg:{:.4f}, accuracy:{:.2f}%'.format(running_loss/len(train_loader.dataset), 100.*correct/len(train_loader.dataset)))
    # record results
    fidout.write('-'*30)
    fidout.write('\nTrain: loss_avg:{:.4f}, accuracy:{:.2f}%'.format(running_loss/len(train_loader.dataset), 100.*correct/len(train_loader.dataset)))
    fidout.flush()


def test(model, device, test_loader, epoch):
    model.eval()
    running_loss = 0
    test_loss = 0
    correct = 0
    precision = Precision()
    recall = Recall()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # start testing
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.float()
            target = target.unsqueeze(1)
            test_loss = F.binary_cross_entropy(torch.sigmoid(output), target)
            
            # get predicted result
            pred = output.round()

            # analysis
            precision.update((pred, target))
            recall.update((pred, target))
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += test_loss.item()

    print('Test: loss_avg:{:.4f}, accuracy:{:.2f}%'.format(running_loss/len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
    print('Precision: {:.4f}'.format(precision.compute()))
    print('Recall: {:.4f}'.format(recall.compute()))
    print('-'*50)

    # record results
    fidout.write('\nTest: loss_avg:{:.4f}, accuracy:{:.2f}%'.format(running_loss/len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
    fidout.write('\nPrecision: {:.4f}'.format(precision.compute()))
    fidout.write('\nRecall: {:.4f}\n'.format(recall.compute()))
    fidout.flush()


def sample_loader(path):
    arr = np.load(path)
    new_arr = np.expand_dims(arr, axis=0)
    return new_arr


def main():
    # training settings
    params = {
        'train_batch': 16, # input batch size for training (default: 16)
        'test_batch': 16, # input batch size for testing (default: 16)
        'epochs': 50, # number of epochs to train (default: 100)
        'learning_rate': 0.05, # learning rate (default: 0.05)
        'gamma': 0.1, # learning rate step gamma (default: 0.1)
        'step_size': 30, # period of learning rate decay (default: 30)
        'model': 1,
        'interval': 500 # interval print one time 
    }

    device = torch.device("cuda")
    transform = transforms.Compose([transforms.ToTensor()])

    # load training data
    train_data = datasets.DatasetFolder(path + 'safety_estimator/train_data_town5_no_new_scenario_6_6', transform=transform, extensions=".npy", loader=sample_loader)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['train_batch'], shuffle=True)

    # load test data
    test_data = datasets.DatasetFolder(path + 'safety_estimator/test_data_town5_no_new_scenario', transform=transform, extensions=".npy", loader=sample_loader)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['test_batch'], shuffle=True)

    # load model parameter
    if params['model'] == 0:
        model = Net0().to(device)
    elif params['model'] == 1:
        model = Net1().to(device)
    else:
        print('error: Unknown model')
    print('model structure: {}'.format(model))
    # record model structure
    fidout.write('model structure: {}\n'.format(model))
    fidout.write('params: {}\n'.format(params))

    # set optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=params['learning_rate'])
    scheduler = StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])

    # start training
    for epoch in range(1, params['epochs'] + 1):
        train(params, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()
    
    # save trained model
    torch.save(model.state_dict(), path + 'models/' + init_time + 'pretrained_safety_model_town05_ANN.pt')

if __name__ == '__main__':
    main()