import os
from os import walk
import getpass
import shutil
import re

carla_version = 'CARLA_0.9.10.1'
address1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
# absolute path
path_train_A = address1 + 'data_for_model/10frames/1700*5/train_data_town5/classA/'
path_train_B = address1 + 'data_for_model/10frames/1700*5/train_data_town5/classB/'
path_test_A = address1 + 'data_for_model/10frames/1700*5/test_data_town5/classA/'
path_test_B = address1 + 'data_for_model/10frames/1700*5/test_data_town5/classB/'

# get filename
file_train_A = []
for (dirpath, dirnames, filenames) in walk(path_train_A):
    file_train_A.extend(filenames)

file_train_B = []
for (dirpath, dirnames, filenames) in walk(path_train_B):
    file_train_B.extend(filenames)

file_test_A = []
for (dirpath, dirnames, filenames) in walk(path_test_A):
    file_test_A.extend(filenames)

file_test_B = []
for (dirpath, dirnames, filenames) in walk(path_test_B):
    file_test_B.extend(filenames)

# is there same file in path_train_A and path_test_A
for file in file_test_A:
    for temp in file_train_A:
        if file == temp:
            print('Find one!')

for file in file_test_B:
    for temp in file_train_B:
        if file == temp:
            print('Find one!')
