#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# This code is generate a library of task plans, given POIs.
# -----------------------------------------

'''
input: lanes of eight POIs
output: a library of task plans
one item in the library is [source, dest, lanes, cost, succ, flag_merge]
'''

from __future__ import print_function
import subprocess
import tempfile
import re
import numpy as np
import time
import math
import os
import getpass
import sys
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils
import itertools

# -----------------------------------------
# definite paths
# -----------------------------------------
path1 = utils.root_path() + 'interaction/'
path2 = utils.root_path() + 'task-level/'
path3 = utils.root_path() + 'interaction/setting/'


# -----------------------------------------
# The part is to initialize cost and rest risk value for each lane
# -----------------------------------------
'''
generate three npy files:
lane_id.npy, where lanes_id_unique: [1, 2, ..., n]
lane_cost.npy, where cost: [x, x, ..., x]
lane_risk.npy, where risk: [[x, x, ..., x], [x, x, ..., x], ..., [x, x, ..., x]]
'''
# load the map, whose format: road id, lane id, x, y, yaw, id (our defined)
# filein = open(path2 + 'all_waypoints_list_sorted_previous.txt', 'r')
raw_data = np.loadtxt(path2 + 'all_waypoints_list_sorted_previous.txt', delimiter='  ')

# -----------------------------------------
# get lane ids
# -----------------------------------------
lanes_id = []
for line in raw_data:
    lanes_id.append(int(line[5]))
lanes_id_unique = utils.unique_list(lanes_id)
print('-'*30)
print('number of lanes id: {}'.format(len(lanes_id)))
# print('lanes id (unique): {}'.format(lanes_id_unique))
# print('lanes id length: {}'.format(len(lanes_id_unique)))
np.save(path3 + 'lane_id.npy', lanes_id_unique)

# -----------------------------------------
# get the cost of each lane
# -----------------------------------------
result = Counter(lanes_id)
# print('result: {}'.format(result))
# print('lane 37: {}'.format(result[37]))
cost = []
for lane_id in lanes_id_unique:
    cost.append(int(result[lane_id]))
print('-'*30)
print('cost: {}'.format(cost))
print('cost length: {}'.format(len(cost)))
np.save(path3 + 'lane_cost.npy', cost)

# -----------------------------------------
# set initial risk of each lane
# -----------------------------------------
risk = []
for lane_id in lanes_id_unique:
    # change left or change right
    risk.append(0)
print('-'*30)
print('risk: {}'.format(risk))
print('risk length: {}'.format(len(risk)))
np.save(path1 + 'lane_risk.npy', risk)

# ----------------------------------------------------------------------------------

# -----------------------------------------
# This part is generate a library of task plans, given POIs.
# -----------------------------------------
'''
input: lanes of eight POIs
output: a library of task plans
one item in the library is [source, dest, lanes, cost, succ, flag_merge]
'''
# -----------------------------------------
# load POIs in the request (fixed)
# -----------------------------------------
POIs_lane = np.load(path3 + 'POIs_lane.npy')
init_lane = int(POIs_lane[0])
home_lane = int(POIs_lane[1])
school_lane = int(POIs_lane[2])
gas1_lane = int(POIs_lane[3])
gas2_lane = int(POIs_lane[4])
grocery1_lane = int(POIs_lane[5])
grocery2_lane = int(POIs_lane[6])

POIs_name = {
    init_lane: 'init_lane',
    school_lane: 'school_lane',
    gas1_lane: 'gas1_lane',
    gas2_lane: 'gas2_lane',
    grocery1_lane: 'grocery1_lane',
    grocery2_lane: 'grocery2_lane',
    home_lane: 'home_lane'
}

print('-'*30)
print('init: {}\n home: {}\n school: {}\n gas1: {}\n gas2: {}\n grocery1: {}\n grocery2: {}'.format(init_lane, home_lane, school_lane, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane))
print('-'*30)

# -----------------------------------------
# feasible combinations of a pair of source_lane and dest_lane
# -----------------------------------------
source_dest_lanes = list(itertools.permutations(([init_lane, school_lane, school_lane, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane, home_lane]), 2))
print('source_dest_lanes: {}\n number of permutations: {}'.format(source_dest_lanes, len(source_dest_lanes)))

temp_source_dest_lanes = []
for item in source_dest_lanes:
    # -----------------------------------------
    # filter rule 1: init_lane should not be the destination
    # filter rule 2: gas 1 and 2 should not be in the same combinations
    # filter rule 3: grocery 1 and 2 should not be in the same combinations
    # filter rule 4: home should not be the source
    # -----------------------------------------
    print('-'*30)
    print('item 1 and 2: {}, {}'.format(item[0], item[1]))
    if (init_lane == item[1]) or (home_lane == item[0]):
        print('sequences of init or home are wrong')
    else:
        if (gas1_lane == item[0] and gas2_lane == item[1]) \
        or (gas2_lane == item[0] and gas1_lane == item[1]):
            print('gas 1 and 2 are in the same combinations')
        else:
            if (grocery1_lane == item[0] and grocery2_lane == item[1]) \
            or (grocery2_lane == item[0] and grocery1_lane == item[1]):
                print('grocery 1 and 2 are in the same combinations')
            else:
                temp_source_dest_lanes.append(item)

source_dest_lanes = temp_source_dest_lanes
print('-'*30)
print('after filtering, source_dest_lanes: {}\n number of permutations: {}'.format(source_dest_lanes, len(source_dest_lanes)))
print('-'*30)

# -----------------------------------------
# compute feasible task plans for each permutations
# -----------------------------------------
library_taskplan = []
for item in source_dest_lanes:
    source_lane = item[0]
    dest_lane = item[1]
    print('source_lane: {}, dest_lane: {}'.format(source_lane, dest_lane))
    poi_info = [POIs_name[item[0]], POIs_name[item[1]]]

    with tempfile.TemporaryFile() as tempf:
    # -----------------------------------------
    # compute a task plan via ASP
    # n = 10 stands for the number of steps
    # '-n' '0' stands for output all feasible plans
    # facts.asp, problem.asp, ruleDriving.asp are knowledge generated by human
    # -----------------------------------------
        proc = subprocess.Popen(['clingo', path2 + 'facts.asp', path2 + 'problem.asp', path2 + 'rulesDriving.asp', '-c', 'n=10', '-c', 'x=' + str(source_lane), '-c', 'y=' + str(dest_lane), '-n', '0'], stdout=tempf)
        proc.wait()
        tempf.seek(0)
        for line in tempf:
            line = line.decode('utf-8') # each line is an output plan
            if line.find('inlane') != -1:
                # line format: inlane(37,0) inlane(47,4) stop(4) stop(5) stop(6) stop(7) stop(8) stop(9) inlane(38,1) changeleft(0) inlane(83,2) turnleft(1) turnleft(2) inlane(139,3) turnright(3) stop(10)
            
                # -----------------------------------------
                # process inline
                # -----------------------------------------
                # inline_list format: ['inlane(37,0)', 'inlane(47,4)', 'inlane(38,1)', 'inlane(83,2)', 'inlane(139,3)']
                inlane_list = re.findall(r'inlane[(]\d+[,]\d+[)]', line)

            
                # inlane_list_name_step format: [[37, 0], [38, 1], [83, 2], [139, 3], [47, 4]]
                inlane_list_name_step = []
                for item_inlane in inlane_list:
                    Lane_name_step = re.findall(r'\d+', item_inlane)
                    inlane_list_name_step.append([int(Lane_name_step[0]), int(Lane_name_step[1])])
            
                # need to sort inlane_list_name_step
                inlane_list_name_step = utils.sort_item_in_list(inlane_list_name_step)

                # get task plan
                task_plan = []
                for x in inlane_list_name_step:
                    task_plan.append(x[0])
                print('task_plan:{}'.format(task_plan))

                # -----------------------------------------
                # process actions
                # -----------------------------------------
                changeleft_list = re.findall(r'changeleft[(]\d+[)]', line)
                changeright_list = re.findall(r'changeright[(]\d+[)]', line)
                turnleft_list = re.findall(r'turnleft[(]\d+[)]', line)
                turnright_list = re.findall(r'turnright[(]\d+[)]', line)
                forward_list = re.findall(r'forward[(]\d+[)]', line)
                stop_list = re.findall(r'stop[(]\d+[)]', line)

                action_list = changeleft_list + changeright_list + turnleft_list + turnright_list + forward_list + stop_list

                # action_list_step format: [['changeleft', 0], ['turnleft', 1], ['turnleft', 2], ['turnright', 3], ['stop', 4], ['stop', 5], ['stop', 6], ['stop', 7], ['stop', 8], ['stop', 9], ['stop', 10]]
                action_list_step = []
                for item_action in action_list:
                    Action = re.findall(r'[a-z]+', item_action)
                    Action = ''.join(Action)
                    Step = re.findall(r'\d+', item_action)
                    Step = ''.join(Step)
                    action_list_step.append([Action, int(Step)])
            
                # need to sort action_list_step
                action_list_step = utils.sort_item_in_list(action_list_step)

                # -----------------------------------------
                # load lane_id, lane_cost, lane_risk
                # -----------------------------------------
                lane_id = np.load(path3 + 'lane_id.npy')
                lane_cost = np.load(path3 + 'lane_cost.npy')
                lane_risk = np.load(path1 + 'lane_risk.npy')

                # -----------------------------------------
                # compute cost of a task plan
                # inlane_list_name_step format: [[37, 0], [38, 1], [83, 2], [139, 3], [47, 4]]
                # action_list_step format: [['changeleft', 0], ['turnleft', 1], ['turnleft', 2], ['turnright', 3], ['stop', 4], ['stop', 5], ['stop', 6], ['stop', 7], ['stop', 8], ['stop', 9], ['stop', 10]]
                # -----------------------------------------
                each_plan_cost = 0  # initialize cost value of a task plan
                cost_info = []
                for item_inlane in inlane_list_name_step:
                    # considering lane length
                    each_plan_cost = each_plan_cost + lane_cost[item_inlane[0]-1]
                    cost_info.append(lane_cost[item_inlane[0]-1])
                
                # remove the length of lane 37 or 38
                for index in range(len(action_list_step)):
                    if action_list_step[index][0] == 'changeleft' or action_list_step[index][0] == 'changeright':
                        each_plan_cost = each_plan_cost - lane_cost[inlane_list_name_step[index][0]-1]

                # considering turning left and right
                cost_turn = 25
                each_plan_cost = each_plan_cost + (len(turnleft_list) + len(turnright_list)) * cost_turn

                # -----------------------------------------
                # compute if changeleft or changeright exists in task plan
                # inlane_list_name_step format: [[37, 0], [38, 1], [83, 2], [139, 3], [47, 4]]
                # action_list_step format: [['changeleft', 0], ['turnleft', 1], ['turnleft', 2], ['turnright', 3], ['stop', 4], ['stop', 5], ['stop', 6], ['stop', 7], ['stop', 8], ['stop', 9], ['stop', 10]]
                # -----------------------------------------
                flag_merge = [] # if the car needs to merge lane
                each_plan_risk = 0  # risk value of a task plan

                for index in range(len(inlane_list_name_step)):
                    lane_name = inlane_list_name_step[index][0]
                    lane_action = action_list_step[index][0]
                    if lane_action == 'changeleft' or lane_action == 'changeright':
                        risk = lane_risk[lane_name-1]
                        each_plan_risk = each_plan_risk + utils.risk_transform(risk)
                        flag_merge.append(1)
                        # print('lane_action:{} flag:{}'.format(lane_action, 1))
                    else:
                        flag_merge.append(0)
            
                # finally, add a task plan info to the library
                library_taskplan.append([item[0], item[1], task_plan, each_plan_cost, each_plan_risk, flag_merge, poi_info, cost_info])
                print('source: {}\n dest: {}\n task plan: {}\n each plan cost: {}\n each_plan_risk: {}\n flag_merge: {}\n poi_info: {}\n cost_info: {}\n'.format(source_lane, dest_lane, task_plan, each_plan_cost, each_plan_risk, flag_merge, poi_info, cost_info))
                
print('library_taskplan is done, whose length is {}'.format(len(library_taskplan)))
np.save(path3 + 'library_taskplan.npy', library_taskplan)