import os
import getpass
import re
import random
import numpy as np

# from matplotlib import pyplot as plt

constant = 4000

address0 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/'
address1 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/'
address2 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/task-level/'
address3 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/setting/'
address4 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/results/'
address6 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/no_collision/'
address7 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/collision/'
address8 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/our/collision/'
address9 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/our/no_collision/'
address10 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/th-based-0.3/collision/'
address11 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/th-based-0.3/no_collision/'
address12 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/th-based-0.8/collision/'
address13 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/th-based-0.8/no_collision/'

home_lane = np.load(address3 + 'home_lane.npy')
gas1_lane = np.load(address3 + 'gas1_lane.npy')
gas2_lane = np.load(address3 + 'gas2_lane.npy')
gas3_lane = np.load(address3 + 'gas3_lane.npy')
grocery1_lane = np.load(address3 + 'grocery1_lane.npy')
grocery2_lane = np.load(address3 + 'grocery2_lane.npy')
school_lane = np.load(address3 + 'school_lane.npy')


def soft_require(source2, source3, source4, gas2_lane, grocery2_lane):
    source_list = [source2, source3, source4]
    source_list = [str(element) for element in source_list]

    soft_penalty1 = 200
    soft_penalty2 = 200

    gas_penalty = 0
    grocery_penalty = 0

    if str(gas2_lane) in source_list:
        gas_penalty = soft_penalty1

    if str(grocery2_lane) in source_list:
        grocery_penalty = soft_penalty2

    return gas_penalty + grocery_penalty

result = [37, 21, 22, 136, 135, 107, 53, 145, 20, 156, 155, 129, 130, 126, 125, 11, 12, 8, 4, 106, 80, 73, 106, 105, 139, 47]
# result = [37, 21, 22, 136, 135, 107, 53, 145, 20, 156, 131, 135, 107, 53, 57, 61, 16, 12, 8, 4, 106, 80, 73, 106, 105, 139, 47]
# result = [37, 21, 22, 136, 135, 107, 53, 145, 20, 156, 131, 135, 107, 53, 57, 61, 36, 126, 12, 8, 4, 106, 80, 73, 106, 105, 139, 47]
print('result:', result)

distance = 0
numLane = np.load(address1 + 'numLane.npy')
for lane in result:
    distance += numLane[int(lane) - 1]
print('distance:', distance)

penalty = 0
for item in result:
    if str(item) == str(gas2_lane):
        penalty += 200
    if str(item) == str(grocery2_lane):
        penalty += 200
print('penalty:', penalty)

turns = np.load('turns.npy')
turn = 0
for index in range(len(result) - 1):
    lane1 = result[index]
    lane2 = result[index + 1]
    for item in turns:
        if int(item[0]) == int(lane1) and int(item[1]) == int(lane2):
            turn += 1
        else:
            continue
print('turn:', turn * 25)

print('total cost:', distance + penalty + turn * 25)
