import os
import getpass
import re
import shutil

address0 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/'
address1 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/'
address2 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/task-level/'
address3 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/setting/'
address4 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/results/'
address5 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/'
address6 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/our/'
address7 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/our/no_collision/'
address8 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/our/collision/'
if not os.path.exists(address6):
    os.mkdir(address6)
if not os.path.exists(address7):
    os.mkdir(address7)
if not os.path.exists(address8):
    os.mkdir(address8)

num = 1000

num_no_collision = 0
num_collision = 0

for i in range(num):
    output_file = 'our_result' + '_' + str(i) + '.txt'
    if os.path.exists(address5 + output_file):
        filein = open(address5 + output_file, 'r')
        for line in filein.readlines():
            if 'distance:' in line:
                distance = re.findall(r'\d+', line)
                print('filename and distance', output_file, distance)
                num_no_collision += 1
                # move this file to no_collision folder
                shutil.move(address5 + output_file, address7 + output_file)
            elif 'collision' in line:
                num_collision += 1
                # move this file to collision folder
                shutil.move(address5 + output_file, address8 + output_file)

print('no collision', num_no_collision)
print('collision', num_collision)


