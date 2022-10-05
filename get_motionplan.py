# -----------------------------------------
# input: the optimal task plan,
# output: motion plan
# -----------------------------------------
# -*- coding: UTF-8 -*-

from __future__ import print_function
import numpy as np
import time
import math
import os
import getpass
import utils

def get_motionplan(root_path):
    # -----------------------------------------
    # definite paths
    # -----------------------------------------
    path1 = root_path + 'interaction/'
    path2 = root_path + 'task-level/'
    path3 = root_path + 'interaction/setting/'

    init_time = time.time()  # to compute time cost of grounding
    
    # -----------------------------------------
    # set parameters
    # -----------------------------------------
    params = {
        'print': False
    }
    
    # -----------------------------------------
    # load data
    # -----------------------------------------
    # load optimal task plan
    optm_plan = np.load(path1 + 'task_plan.npy')
    # load lane_cost.npy and wayPoints.npy
    lane_cost = np.load(path3 + 'lane_cost.npy')
    if os.path.exists(path3 + 'wayPoints.npy'):
        wayPoints = np.load(path3 + 'wayPoints.npy')
    else:
        wayPoints = np.loadtxt(path2 + 'all_waypoints_list_sorted_previous.txt', delimiter='  ')
        np.save(path3 + 'wayPoints.npy', wayPoints)

    # -----------------------------------------
    # start grounding
    # -----------------------------------------
    coords = []  # one selected waypoint of each lane
    Endpoints = []
    for lane in optm_plan[0]:
        for i in range(0, len(wayPoints)):
            if wayPoints[i][5] == lane:
                lane_length = lane_cost[int(wayPoints[i][5]) - 1]
                forward_endpoint = wayPoints[i + round(int(lane_length) / 4)]
                after_endpoint = wayPoints[i + round(int(lane_length) / 4 * 3)]
                Endpoints.append(forward_endpoint)
                Endpoints.append(after_endpoint)
                break


    first_distance = utils.compute_dist(Endpoints[0][2], Endpoints[0][3], Endpoints[2][2], Endpoints[2][3])
    second_distance = utils.compute_dist(Endpoints[1][2], Endpoints[1][3], Endpoints[2][2], Endpoints[2][3])
    # two situations: 1: not parallel lanes 2: parallel lanes
    if first_distance > 5 and second_distance > 5:
        if first_distance > second_distance:
            start_loc = Endpoints[0]
        else:
            start_loc = Endpoints[1]
    elif first_distance < 5 or second_distance < 5:
        first_distance = utils.compute_dist(Endpoints[0][2], Endpoints[0][3], Endpoints[4][2], Endpoints[4][3])
        second_distance = utils.compute_dist(Endpoints[1][2], Endpoints[1][3], Endpoints[4][2], Endpoints[4][3])
        if first_distance > second_distance:
            start_loc = Endpoints[0]
        else:
            start_loc = Endpoints[1]
    else:
        print('Error!')

    coords.append((start_loc[2], start_loc[3], start_loc[4]))
    for i in range(2, len(Endpoints), 2):
        last_choice_i = i

    for i in range(2, len(Endpoints), 2):
        forward_endpoint = Endpoints[i]
        after_endpoint = Endpoints[i + 1]
        first_distance = utils.compute_dist(forward_endpoint[2], forward_endpoint[3], start_loc[2], start_loc[3])
        second_distance = utils.compute_dist(after_endpoint[2], after_endpoint[3], start_loc[2], start_loc[3])
        if optm_plan[1][int(i / 2.0) - 1] != 1:
            # need merge lane
            if first_distance < second_distance:
                start_loc = forward_endpoint
                if i == last_choice_i:
                    start_loc = after_endpoint
            else:
                start_loc = after_endpoint
                if i == last_choice_i:
                    start_loc = forward_endpoint
        else:
            # not need merge lane
            if first_distance < second_distance:
                start_loc = after_endpoint
                if i == last_choice_i:
                    start_loc = forward_endpoint
            else:
                start_loc = forward_endpoint
                if i == last_choice_i:
                    start_loc = after_endpoint
        coords.append((start_loc[2], start_loc[3], start_loc[4]))

    # save coordinates
    site = [] # save int type
    trajectory = [] # save float type
    action_index = 0
    for pos in coords:
        site.append([int(action_index), int(optm_plan[1][action_index]), int(optm_plan[0][action_index])])
        trajectory.append([float(pos[0]), float(pos[1]), float(pos[2])])
        action_index = action_index + 1
    np.save(path1 + 'site.npy', site)
    np.save(path1 + 'trajectory.npy', trajectory)
    # print('site:{}'.format(site))
    # print('trajectory:{}'.format(trajectory))

    if params['print']:
        print('computing time cost (s):{}\n'.format(time.time() - init_time))
