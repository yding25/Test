#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to rename files in old folder and moves them to new folder.
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
import shutil
import re

# -----------------------------------------
# change a file name
# -----------------------------------------
def changeName(oldName):
    new_name = '107_108_' + oldName
    return new_name

# -----------------------------------------
# create new folders
# -----------------------------------------
carla_version = 'CARLA_0.9.10.1'
path1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
if not os.path.exists(path1 + 'data_for_model/raw_data_full/'):
    os.mkdir(path1 + 'data_for_model/raw_data_full/')
if not os.path.exists(path1 + 'data_for_model/raw_data_full/raw_frame/'):
    os.mkdir(path1 + 'data_for_model/raw_data_full/raw_frame/')

# -----------------------------------------
# set absolute paths
# -----------------------------------------
old_path = path1 + 'data_for_model/raw_data_107_108/raw_frame/'
new_path = path1 + 'data_for_model/raw_data_full/raw_frame/'

# -----------------------------------------
# start changing name
# -----------------------------------------
filename_list = []
for (dirpath, dirnames, filenames) in os.walk(old_path):
    filename_list.extend(filenames)
    for old_name in filename_list:
        new_name = changeName(old_name)
        print('new_name:{}'.format(new_name))
        shutil.copy(old_path + old_name, new_path + new_name)
