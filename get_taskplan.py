#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# This code is to compute an optimal task plan for the request
# -----------------------------------------

'''
request: a mother needs to pick up her kid from the school, go to grocery, and go to gas station
user perfernce: kid does not like the gas smell, but kid likes shopping

given current lane, visited POIs and updated risk data
output the optimal task plan
'''

from __future__ import print_function
import os
import time
import getpass
import math
import numpy as np
import shutil
import utils
import itertools

def get_taskplan(root_path, risk_alpha, penalty):
    # -----------------------------------------
    # definite paths
    # -----------------------------------------
    path1 = root_path + 'interaction/'
    path2 = root_path + 'task-level/'
    path3 = root_path + 'interaction/setting/'
    path4 = root_path + 'interaction/results/'

    init_time = time.time()

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

    # -----------------------------------------
    # set parameters
    # -----------------------------------------
    params = {
        'print': False
    }

    # -----------------------------------------
    # load current states
    # -----------------------------------------
    curr_lane = np.load(path1 + 'current_lane.npy')
    curr_lane = int(curr_lane)
    visited_milestones = np.load(path1 + 'visited_milestones.npy')
    lane_risk = np.load(path1 + 'lane_risk.npy')

    # -----------------------------------------
    # all general task plan (six kinds)
    # -----------------------------------------
    g_taskplan = [['gas', 'school', 'grocery'],
                        ['gas', 'grocery', 'school'],
                        ['school', 'gas', 'grocery'],
                        ['school', 'grocery', 'gas'],
                        ['grocery', 'school', 'gas'],
                        ['grocery', 'gas', 'school']]

    # -----------------------------------------
    # analyze current states: how many pois (except init_lane and home_lane) are visited and which?
    # -----------------------------------------
    visited = []
    for item in visited_milestones:
        if item in [gas1_lane, gas2_lane]:
            visited.append('gas')
        elif item in [school_lane]:
            visited.append('school')
        elif item in [grocery1_lane, grocery2_lane]:
            visited.append('grocery')
        else:
            print('unknown item\t item:{}\t visited_milestones:{}'.format(item, visited_milestones))

    if params['print']:
        print('current lane: {}\n visited milestones (lane id):{}\n visited POIs (kind name): {}'.format(curr_lane, visited_milestones, visited))

    # -----------------------------------------
    # select feasible general task plan according to visited
    # -----------------------------------------
    # only parts of g_taskplan meet requirements (visited)
    fsib_g_taskplan = utils.compatible(g_taskplan, visited)
    if params['print']:
        print('feasible general taskplan: {}'.format(fsib_g_taskplan))

    # -----------------------------------------
    # compuete all feasible task plans according to feasible general taskplan and visited_milestones
    # -----------------------------------------
    taskplan_milestone = []
    for gplan in fsib_g_taskplan:
        # lane of the first poi
        first_poi_lane = utils.first_poi_2_lane(gplan[0], visited_milestones, init_lane, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane, school_lane)
        
        # lane of the last poi: home_lane

        # lane of the middle pois
        mid_pois = gplan[1:len(gplan)-1]
        if len(mid_pois) > 0: # if mid poi exists
            mid_pois_lane = []
            # poi name -> poi lane
            for poi in mid_pois:
                mid_pois_lane.append(utils.mid_poi_2_lane(poi, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane, school_lane))
            # start producting
            if len(mid_pois_lane) == 1:
                # if only one mid poi
                for item in list(itertools.product(mid_pois_lane[0])):
                    taskplan_milestone.append([first_poi_lane, item[0], home_lane])
            elif len(mid_pois_lane) == 2:
                # if two mid poi
                for item in list(itertools.product(mid_pois_lane[0], mid_pois_lane[1])):
                    taskplan_milestone.append([first_poi_lane, item[0], item[1], home_lane])
            elif len(mid_pois_lane) == 3:
                # if three mid poi
                for item in list(itertools.product(mid_pois_lane[0], mid_pois_lane[1], mid_pois_lane[2])):
                    taskplan_milestone.append([first_poi_lane, item[0], item[1], item[2], home_lane])
            else:
                print('Error: wrong number of mid pois')
        else:
            taskplan_milestone = [[first_poi_lane, home_lane]]
    if params['print']:
        print('feasible task plans (milestones): {}'.format(taskplan_milestone))

    # -----------------------------------------
    # load library_taskplan
    # -----------------------------------------
    library_taskplan = np.load(path3 + 'library_taskplan.npy', allow_pickle=True)

    # -----------------------------------------
    # get the optimal task plan with details (pois in sequence, corresponding milestones, plan, flag of merge,
    # cost, risk, pref, overall utility etc)
    # -----------------------------------------
    optm_pois = []
    optm_milestone = []
    optm_plan = []
    optm_flag_merge = []
    optm_utility = -99999
    optm_plan_cost = 0
    optm_plan_risk = 0
    optm_plan_pref = 0
    for plan_milestone in taskplan_milestone:
        '''
        example: taskplan_milestone = [[37, 38, 104, 56, 47],
                            [37, 38, 104, 47],
                            [37, 38, 47],
                            [37, 47]]
        37: init_lane
        47: home_lane
        '''
        if len(plan_milestone) <= 1:
            print('Error: wrong length of plan_milestone')
        elif len(plan_milestone) == 2: 
            # -----------------------------------------
            # example: plan_milestone=[37, 47]
            # -----------------------------------------
            lanes = plan_milestone
            fsib_items_lanes_1_2 = []
            for item in library_taskplan:
                '''
                item[0]: source_lane
                item[1]: dest_lane
                items[2]: task plan
                items[3]: plan_cost
                items[4]: plan_risk
                items[5]: flag_merge
                items[6]: poi_names
                items[7]: cost of each lane
                '''
                # find all feasible detailed task plan between two lanes according to the library_taskplan
                if lanes[0] == item[0] and lanes[1] == item[1] and curr_lane in item[2]: # current_lane should be in task plan between lane 1 and lane 2
                    fsib_items_lanes_1_2.append(item)

            if len(fsib_items_lanes_1_2) > 0:
                # -----------------------------------------
                # output the optimal task plan with minmal cost and risk between two lanes
                # -----------------------------------------
                taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2 = utils.min_cost_risk_2lanes(fsib_items_lanes_1_2, risk_alpha, lane_risk)
                if params['print']:
                    print('-'*30)
                    print('current plan (milestone):\n taskplan lanes 1->2:{}\n cost+risk:{}, cost:{}, risk:{}'.format(taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2))
                
                # compute the cost and risk among all lanes
                cost = cost_lanes_1_2
                risk = risk_lanes_1_2

                # -----------------------------------------
                # compute user preference for plan (milestone)
                # -----------------------------------------
                pref = 0
                
                if params['print']:
                    print('plan cost+risk+pref:{}\n cost: {}\n plan risk:{}\n plan preference:{}\n'.format(cost + risk + pref, cost, risk, pref))
                
                if optm_utility <= -(cost + risk + pref):
                    optm_pois = taskplan_lanes_1_2[6]
                    optm_milestone = plan_milestone
                    optm_plan = taskplan_lanes_1_2[2]
                    optm_flag_merge = taskplan_lanes_1_2[5]
                    optm_plan_cost = cost
                    optm_plan_risk = risk
                    optm_plan_pref = pref
                    optm_utility = -(cost + risk + pref)
            else:
                if params['print']:
                    print('Skip this plan_milestone, because one task plan between two lanes is empty')
                    print('Details:\n lanes 1->2:{}'.format(len(fsib_items_lanes_1_2)))

        elif len(plan_milestone) == 3: 
            # -----------------------------------------
            # example: plan_milestone=[37, 38, 47]
            # -----------------------------------------
            lanes = plan_milestone
            fsib_items_lanes_1_2 = []
            fsib_items_lanes_2_3 = []
            for item in library_taskplan:
                '''
                item[0]: source_lane
                item[1]: dest_lane
                items[2]: task plan
                items[3]: plan_cost
                items[4]: plan_risk
                items[5]: flag_merge
                items[6]: poi_names
                items[7]: cost of each lane
                '''
                # find all feasible detailed task plan between two lanes according to the library_taskplan
                if lanes[0] == item[0] and lanes[1] == item[1] and curr_lane in item[2]: # current_lane should be in task plan between lane 1 and lane 2
                    fsib_items_lanes_1_2.append(item)
                if lanes[1] == item[0] and lanes[2] == item[1]:
                    fsib_items_lanes_2_3.append(item)

            if len(fsib_items_lanes_1_2) > 0 and len(fsib_items_lanes_2_3) > 0:
                # -----------------------------------------
                # output the optimal task plan with minmal cost and risk between two lanes
                # -----------------------------------------
                taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2 = utils.min_cost_risk_2lanes(fsib_items_lanes_1_2, risk_alpha, lane_risk)
                taskplan_lanes_2_3, cost_risk_lanes_2_3, cost_lanes_2_3, risk_lanes_2_3 = utils.min_cost_risk_2lanes(fsib_items_lanes_2_3, risk_alpha, lane_risk)
                if params['print']:
                    print('-'*30)
                    print('current plan (milestone):\n taskplan lanes 1->2:{}\n cost+risk:{}, cost:{}, risk:{}\n taskplan lanes 2->3:{}\n cost+risk:{}, cost:{}, risk:{}'.format(taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2, taskplan_lanes_2_3, cost_risk_lanes_2_3, cost_lanes_2_3, risk_lanes_2_3))
                
                # compute the cost and risk among all lanes
                cost = cost_lanes_1_2 + cost_lanes_2_3
                risk = risk_lanes_1_2 + risk_lanes_2_3

                # -----------------------------------------
                # compute user preference for plan (milestone)
                # -----------------------------------------
                taskplan_lanes = [taskplan_lanes_1_2, taskplan_lanes_2_3]
                pref = utils.preference_lanes(penalty, taskplan_lanes)
                
                if params['print']:
                    print('plan cost+risk+pref:{}\n cost: {}\n plan risk:{}\n plan preference:{}\n'.format(cost + risk + pref, cost, risk, pref))
                
                if optm_utility <= -(cost + risk + pref):
                    optm_pois = taskplan_lanes_1_2[6] + taskplan_lanes_2_3[6][1:]
                    optm_milestone = plan_milestone
                    optm_plan = taskplan_lanes_1_2[2] + taskplan_lanes_2_3[2][1:]
                    optm_flag_merge = taskplan_lanes_1_2[5][:-1] + taskplan_lanes_2_3[5]
                    optm_plan_cost = cost
                    optm_plan_risk = risk
                    optm_plan_pref = pref
                    optm_utility = -(cost + risk + pref)
            else:
                if params['print']:
                    print('Skip this plan_milestone, because one task plan between two lanes is empty')
                    print('Details:\n lanes 1->2:{}\n lanes 2->3:{}'.format(len(fsib_items_lanes_1_2), len(fsib_items_lanes_2_3)))
        
        elif len(plan_milestone) == 4:
            # -----------------------------------------
            # example: plan_milestone=[37, 38, 104, 47]
            # -----------------------------------------
            lanes = plan_milestone
            fsib_items_lanes_1_2 = []
            fsib_items_lanes_2_3 = []
            fsib_items_lanes_3_4 = []
            for item in library_taskplan:
                '''
                item[0]: source_lane
                item[1]: dest_lane
                items[2]: task plan
                items[3]: plan_cost
                items[4]: plan_risk
                items[5]: flag_merge
                items[6]: poi_names
                items[7]: cost of each lane
                '''
                # find all feasible detailed task plan between two lanes according to the library_taskplan
                if lanes[0] == item[0] and lanes[1] == item[1] and curr_lane in item[2]: # current_lane should be in task plan between lane 1 and lane 2
                    fsib_items_lanes_1_2.append(item)
                if lanes[1] == item[0] and lanes[2] == item[1]:
                    fsib_items_lanes_2_3.append(item)
                if lanes[2] == item[0] and lanes[3] == item[1]:
                    fsib_items_lanes_3_4.append(item)

            if len(fsib_items_lanes_1_2) > 0 and len(fsib_items_lanes_2_3) > 0 and len(fsib_items_lanes_3_4) > 0:
                # -----------------------------------------
                # output the optimal task plan with minmal cost and risk between two lanes
                # -----------------------------------------
                taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2 = utils.min_cost_risk_2lanes(fsib_items_lanes_1_2, risk_alpha, lane_risk)
                taskplan_lanes_2_3, cost_risk_lanes_2_3, cost_lanes_2_3, risk_lanes_2_3 = utils.min_cost_risk_2lanes(fsib_items_lanes_2_3, risk_alpha, lane_risk)
                taskplan_lanes_3_4, cost_risk_lanes_3_4, cost_lanes_3_4, risk_lanes_3_4 = utils.min_cost_risk_2lanes(fsib_items_lanes_3_4, risk_alpha, lane_risk)
                if params['print']:
                    print('-'*30)
                    print('current plan (milestone):\n taskplan lanes 1->2:{}\n cost+risk:{}, cost:{}, risk:{}\n taskplan lanes 2->3:{}\n cost+risk:{}, cost:{}, risk:{}\n taskplan lanes 3->4:{}\n cost+risk:{}, cost:{}, risk:{}'.format(taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2, taskplan_lanes_2_3, cost_risk_lanes_2_3, cost_lanes_2_3, risk_lanes_2_3, taskplan_lanes_3_4, cost_risk_lanes_3_4, cost_lanes_3_4, risk_lanes_3_4))
                
                # compute the cost and risk among all lanes
                cost = cost_lanes_1_2 + cost_lanes_2_3 + cost_lanes_3_4
                risk = risk_lanes_1_2 + risk_lanes_2_3 + risk_lanes_3_4

                # -----------------------------------------
                # compute user preference for plan (milestone)
                # -----------------------------------------
                taskplan_lanes = [taskplan_lanes_1_2, taskplan_lanes_2_3, taskplan_lanes_3_4]
                pref = utils.preference_lanes(penalty, taskplan_lanes)
                
                if params['print']:
                    print('plan cost+risk+pref:{}\n cost: {}\n plan risk:{}\n plan preference:{}\n'.format(cost + risk + pref, cost, risk, pref))
                
                if optm_utility <= -(cost + risk + pref):
                    optm_pois = taskplan_lanes_1_2[6] + taskplan_lanes_2_3[6][1:] + taskplan_lanes_3_4[6][1:]
                    optm_milestone = plan_milestone
                    optm_plan = taskplan_lanes_1_2[2] + taskplan_lanes_2_3[2][1:] + taskplan_lanes_3_4[2][1:]
                    optm_flag_merge = taskplan_lanes_1_2[5][:-1] + taskplan_lanes_2_3[5][:-1] + taskplan_lanes_3_4[5]
                    optm_plan_cost = cost
                    optm_plan_risk = risk
                    optm_plan_pref = pref
                    optm_utility = -(cost + risk + pref)
            else:
                if params['print']:
                    print('Skip this plan_milestone, because one task plan between two lanes is empty')
                    print('Details:\n lanes 1->2:{}\n lanes 2->3:{}\n lanes 3->4:{}'.format(len(fsib_items_lanes_1_2), len(fsib_items_lanes_2_3), len(fsib_items_lanes_3_4)))
        elif len(plan_milestone) == 5:
            # -----------------------------------------
            # example: plan_milestone=[37, 38, 104, 56, 47]
            # -----------------------------------------
            lanes = plan_milestone
            fsib_items_lanes_1_2 = []
            fsib_items_lanes_2_3 = []
            fsib_items_lanes_3_4 = []
            fsib_items_lanes_4_5 = []
            for item in library_taskplan:
                '''
                item[0]: source_lane
                item[1]: dest_lane
                items[2]: task plan
                items[3]: plan_cost
                items[4]: plan_risk
                items[5]: flag_merge
                items[6]: poi_names
                items[7]: cost of each lane
                '''
                # find all feasible detailed task plan between two lanes according to the library_taskplan
                if lanes[0] == item[0] and lanes[1] == item[1] and curr_lane in item[2]: # current_lane should be in task plan between lane 1 and lane 2
                    fsib_items_lanes_1_2.append(item)
                if lanes[1] == item[0] and lanes[2] == item[1]:
                    fsib_items_lanes_2_3.append(item)
                if lanes[2] == item[0] and lanes[3] == item[1]:
                    fsib_items_lanes_3_4.append(item)
                if lanes[3] == item[0] and lanes[4] == item[1]:
                    fsib_items_lanes_4_5.append(item)
            if len(fsib_items_lanes_1_2) > 0 and len(fsib_items_lanes_2_3) > 0 and len(fsib_items_lanes_3_4) > 0 and len(fsib_items_lanes_4_5) > 0:
                # -----------------------------------------
                # output the optimal task plan with minmal cost and risk between two lanes
                # -----------------------------------------
                taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2 = utils.min_cost_risk_2lanes(fsib_items_lanes_1_2, risk_alpha, lane_risk)
                taskplan_lanes_2_3, cost_risk_lanes_2_3, cost_lanes_2_3, risk_lanes_2_3 = utils.min_cost_risk_2lanes(fsib_items_lanes_2_3, risk_alpha, lane_risk)
                taskplan_lanes_3_4, cost_risk_lanes_3_4, cost_lanes_3_4, risk_lanes_3_4 = utils.min_cost_risk_2lanes(fsib_items_lanes_3_4, risk_alpha, lane_risk)
                taskplan_lanes_4_5, cost_risk_lanes_4_5, cost_lanes_4_5, risk_lanes_4_5 = utils.min_cost_risk_2lanes(fsib_items_lanes_4_5, risk_alpha, lane_risk)
                if params['print']:
                    print('-'*30)
                    print('current plan (milestone):\n taskplan lanes 1->2:{}\n cost+risk:{}, cost:{}, risk:{}\n taskplan lanes 2->3:{}\n cost+risk:{}, cost:{}, risk:{}\n taskplan lanes 3->4:{}\n cost+risk:{}, cost:{}, risk:{}\n taskplan lanes 4->5:{}\n cost+risk:{}, cost:{}, risk:{}'.format(taskplan_lanes_1_2, cost_risk_lanes_1_2, cost_lanes_1_2, risk_lanes_1_2, taskplan_lanes_2_3, cost_risk_lanes_2_3, cost_lanes_2_3, risk_lanes_2_3, taskplan_lanes_3_4, cost_risk_lanes_3_4, cost_lanes_3_4, risk_lanes_3_4, taskplan_lanes_4_5, cost_risk_lanes_4_5, cost_lanes_4_5, risk_lanes_4_5))
                
                # compute the cost and risk among all lanes
                cost = cost_lanes_1_2 + cost_lanes_2_3 + cost_lanes_3_4 + cost_lanes_4_5
                risk = risk_lanes_1_2 + risk_lanes_2_3 + risk_lanes_3_4 + risk_lanes_4_5

                # -----------------------------------------
                # compute user preference for plan (milestone)
                # -----------------------------------------
                taskplan_lanes = [taskplan_lanes_1_2, taskplan_lanes_2_3, taskplan_lanes_3_4, taskplan_lanes_4_5]
                pref = utils.preference_lanes(penalty, taskplan_lanes)

                if params['print']:
                    print('plan cost+risk+pref:{}\n cost: {}\n plan risk:{}\n plan preference:{}\n'.format(cost + risk + pref, cost, risk, pref))
                
                if optm_utility <= -(cost + risk + pref):
                    optm_pois = taskplan_lanes_1_2[6] + taskplan_lanes_2_3[6][1:] + taskplan_lanes_3_4[6][1:] + taskplan_lanes_4_5[6][1:]
                    optm_milestone = plan_milestone
                    optm_plan = taskplan_lanes_1_2[2] + taskplan_lanes_2_3[2][1:] + taskplan_lanes_3_4[2][1:] + taskplan_lanes_4_5[2][1:]
                    optm_flag_merge = taskplan_lanes_1_2[5][:-1] + taskplan_lanes_2_3[5][:-1] + taskplan_lanes_3_4[5][:-1] + taskplan_lanes_4_5[5]
                    optm_plan_cost = cost
                    optm_plan_risk = risk
                    optm_plan_pref = pref
                    optm_utility = -(cost + risk + pref)
            else:
                if params['print']:
                    print('Skip this plan_milestone, because one task plan between two lanes is empty')
                    print('Details:\n lanes 1->2:{}\n lanes 2->3:{}\n lanes 3->4:{}\n lanes 4->5:{}'.format(len(fsib_items_lanes_1_2), len(fsib_items_lanes_2_3), len(fsib_items_lanes_3_4), len(fsib_items_lanes_4_5)))
        else:
            print('Error: length of a plan is wrong')

    if params['print']:
        print('*'*30)
        print('optimal plan (full):\n POIs name:{}\n milestones:{}\n plan:{}\n flage of merge:{}\n cost:{}\n risk:{}\n preference:{}\n utility;{}'.format(optm_pois, optm_milestone, optm_plan, optm_flag_merge, optm_plan_cost, optm_plan_risk, optm_plan_pref, optm_utility))

    # according to the visited milestone and current lane, output the remaining task plan
    remain_taskplan, remain_flage_merge = utils.generate_remain_taskplan(visited_milestones, curr_lane, optm_plan, optm_flag_merge)

    if params['print']:
        print('-'*30)
        print('optimal plan (remaining):\n plan:{}\n flage of merge:{}'.format(remain_taskplan, remain_flage_merge))
        print('-'*30)

    # save optimal task plan
    np.save(path1 + 'task_plan.npy', [remain_taskplan, remain_flage_merge])

    # save milestone
    np.save(path1 + 'optm_milestone.npy', optm_milestone)

    if params['print']:
        print('computing time cost (s):{}\n'.format(time.time() - init_time))