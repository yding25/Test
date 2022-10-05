import os
from os import walk
import getpass
import shutil
import re
import numpy as np


carla_version = 'CARLA_0.9.10.1'
address1 = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
# absolute path
path = address1 + 'data_for_model/raw_data/raw_png/'

# get filename
file = []
for (dirpath, dirnames, filenames) in walk(path):
    file.extend(filenames)

# record small files
file_small = []
for file in filenames:
    fileSize = os.path.getsize(path+file)
    if fileSize < 100:
        file_small.append(file[0:10])

# remove repeated files
file_small_qualified = []
for temp in file_small:
    if temp not in file_small_qualified:
        file_small_qualified.append(temp)
file_small = file_small_qualified
print(file_small_qualified)

np.save('smallFile.npy', file_small)

fidin = open(address1 + 'data_for_model/raw_data/overview.txt', 'r')
fidin_new = open(address1 + 'data_for_model/raw_data/overview_removed.txt', 'w')
# start classifying
for line in fidin.readlines():
    line = line.strip()
    line = line.split(':')
    filename = line[1]
    filename = filename[1:]
    signal_found = 0
    for temp in file_small:
        if temp == filename:
            signal_found = 1
            print('find')
            break
    if signal_found == 0:
        a = line[0] + ': ' + line[1]
        fidin_new.write(a)
        fidin_new.write('\n')
fidin_new.close()


