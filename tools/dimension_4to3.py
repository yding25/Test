import os
from os import walk
import getpass
import shutil
import re
import numpy as np


carla_version = 'CARLA_0.9.10.1'
address1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
# absolute path
path = address1 + 'data_for_model/raw_data/raw_frame/'

new_path = address1 + 'data_for_model/raw_data/raw_frame_back/'
# get filename
file = []
for (dirpath, dirnames, filenames) in walk(path):
    file.extend(filenames)

for name in file:
    try:
        temp = np.load(path + name)
        print(temp)
        front = temp[0:8192]
        back = temp[8192:8192*2]
        right = temp[8192*2:8192*3]
        left = temp[8192*3:8192*4]
        new_frame = np.concatenate((front, back, left))
        # print(len(new_frame))
        np.save(new_path + name, new_frame)
    except:
        print('dsds')