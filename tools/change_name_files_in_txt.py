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
    new_name = oldName.replace('165', '107_108_165', 1)
    return new_name

# -----------------------------------------
# create new folders
# -----------------------------------------
carla_version = 'CARLA_0.9.10.1'
path1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
if not os.path.exists(path1 + 'data_for_model/raw_data_full/'):
    os.mkdir(path1 + 'data_for_model/raw_data_full/')

# -----------------------------------------
# set absolute paths
# -----------------------------------------
old_path = path1 + 'data_for_model/raw_data_107_108/overview.txt'
new_path = path1 + 'data_for_model/raw_data_full/overview.txt'

# -----------------------------------------
# start changing name
# -----------------------------------------
fid_old = open(old_path, 'r')
fid_new = open(new_path, 'a')

for old_name in fid_old.readlines():
    print('old_name:{}'.format(old_name))
    new_name = changeName(old_name)
    print('new_name:{}'.format(new_name))
    fid_new.write(new_name)