#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to collect data!
# -----------------------------------------

from __future__ import print_function
import argparse
import glob
import logging
import os
import sys
import getpass
import weakref
import cv2
from queue import Queue, Empty
import numpy as np
import time
import math
import random
import torch
from collections import deque
try:
    # -----------------------------------------
    # find carla module
    # -----------------------------------------
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    # -----------------------------------------
    # find carla module (local)
    # -----------------------------------------
    sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/carla')
    # -----------------------------------------
    # add PythonAPI for release mode
    # -----------------------------------------
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    pass
import carla
from carla import ColorConverter
from agents.navigation.basic_agent import BasicAgent
from models.model_supervised import Model_Segmentation_Traffic_Light_Supervised
import re

# -----------------------------------------
# import customized functions
# -----------------------------------------
from utils import compute_direction, generate_navigation

global path1
carla_version = 'CARLA_0.9.10.1'
path1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'

class SupervisedEncoder:
    def __init__(self):
        self.steps_image = [-10, -2, -1, 0]
        self.model_supervised = Model_Segmentation_Traffic_Light_Supervised(len(self.steps_image),
                                                                            len(self.steps_image), 1024, 6, 4,
                                                                            crop_sky=False)
        self.device = torch.device("cuda")
        path_to_encoder_dict = path1 + 'interaction/model_epoch_34.pth'
        self.model_supervised.load_state_dict(torch.load(path_to_encoder_dict, map_location=self.device))
        self.model_supervised.to(device=self.device)
        self.encoder = self.model_supervised.encoder
        self.last_conv_downsample = self.model_supervised.last_conv_downsample
        self._rgb_queue = None
        self.window = (max([abs(number) for number in self.steps_image]) + 1)
        self.RGB_image_buffer = deque([], maxlen=self.window)
        for _ in range(self.window):
            self.RGB_image_buffer.append(np.zeros((3, 288, 288)))
        self.render = True

    def carla_img_to_np(self, carla_img):
        carla_img.convert(ColorConverter.Raw)
        img = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_img.height, carla_img.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]  # carla uses bgr so need to convert it to rgb
        return img

    def run_step(self, np_array_RGB_input, name, snapshot, save_flag=False):
        if save_flag:
            for indice_image in self.steps_image:
                new_image = self.RGB_image_buffer[indice_image + self.window - 1].copy()
                new_image = np.rollaxis(new_image, 0, 3)
                new_bgr = new_image[:, :, ::-1]
                cv2.imwrite(
                    path1 + 'data_for_model/raw_data_41_42_lowGPU/raw_png/' + str(init_time) + '_' + snapshot + "_" + str(
                        abs(indice_image)) + "_" + name + ".png",
                    new_bgr)

        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input)
                .to(dtype=torch.float32, device=self.device)
                .div_(255)
                .unsqueeze(0)
        )

        with torch.no_grad():
            current_encoding = self.encoder(torch_tensor_input)
            current_encoding = self.last_conv_downsample(current_encoding)

        current_encoding_np = current_encoding.cpu().numpy().flatten()
        return current_encoding_np

def main():
    # -----------------------------------------
    #  add four encoders
    # -----------------------------------------
    supervised_encoder_front = SupervisedEncoder()
    supervised_encoder_back = SupervisedEncoder()
    supervised_encoder_right = SupervisedEncoder()
    supervised_encoder_left = SupervisedEncoder()

    def parse(contex):
        items = re.split('_', contex)
        time = items[0]
        name = items[1]
        snapshot = items[2].replace('.npy', '')
        # print('contex: {}, time: {}, name: {}, snapshot: {}'.format(contex, time, name, snapshot))
        return time, name, snapshot

    # -----------------------------------------
    # get filename
    # -----------------------------------------
    filename_list = []
    current_path = path1 + 'data_for_model/raw_data_41_42_lowGPU/encoded_png/'
    for (dirpath, dirnames, filenames) in os.walk(current_path):
        # print('filenames: %s' % filenames)
        for item in filenames:
            if '.npy' in item:
                time, name, snapshot = parse(item)
                # print('item: {}, time: {}, name: {}, snapshot: {}'.format(time, time, name, snapshot))
                filename_list.append([item, time, name, snapshot])

    # -----------------------------------------
    # encode filename
    # -----------------------------------------
    flag_front = False
    flag_back = False
    flag_right = False
    flag_left = False
    for item in filename_list:
        filename = item[0]
        try:
            np_array_RGB_input = np.load(path1 + 'data_for_model/raw_data_41_42_lowGPU/encoded_png/' + filename, allow_pickle=True)
            init_time = item[1]
            name = item[2]
            snapshot = item[3]

            if name == 'Front':
                front = supervised_encoder_back.run_step(np_array_RGB_input, name, snapshot, save_flag=False)
                flag_front = True
            elif name == 'Back':
                back = supervised_encoder_back.run_step(np_array_RGB_input, name, snapshot, save_flag=False)
                flag_back = True
            elif name == 'Right':
                right = supervised_encoder_back.run_step(np_array_RGB_input, name, snapshot, save_flag=False)
                flag_right = True
            elif name == 'Left':
                left = supervised_encoder_back.run_step(np_array_RGB_input, name, snapshot, save_flag=False)
                flag_left = True
        
            if flag_front and flag_back and flag_right and flag_left:
                frames_data = np.concatenate((front, back, right, left))
                np.save(path1 + 'data_for_model/raw_data_41_42_lowGPU/raw_frame/' + str(init_time) + '_' + str(snapshot), frames_data)
                flag_front = False
                flag_back = False
                flag_right = False
                flag_left = False
        except:
            os.remove(path1 + 'data_for_model/raw_data_41_42_lowGPU/encoded_png/' + filename)


if __name__ == '__main__':
    main()
