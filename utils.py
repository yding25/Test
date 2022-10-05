import math
import time
import numpy as np
import os
import getpass
import random
import matplotlib.pyplot as plt
import itertools


def root_path():
    carla_version = 'CARLA_0.9.10.1'
    path = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/Test/'
    return path

def root_path_NoSafe():
    carla_version = 'CARLA_0.9.10.1'
    path = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/NoSafe/'
    return path

def root_path_NoCost():
    carla_version = 'CARLA_0.9.10.1'
    path = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/NoCost/'
    return path

def root_path_NoPref():
    carla_version = 'CARLA_0.9.10.1'
    path = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/NoPref/'
    return path

def root_path_ThSafe():
    carla_version = 'CARLA_0.9.10.1'
    path = '/home/' + getpass.getuser() + '/' + carla_version + '/PythonAPI/ThSafe/'
    return path
    

def compute_direction(x, y):
    '''
    given two points x and y
    compute the direction of two points, 
    '''
    origin_x = x[0]
    origin_y = x[1]
    destination_x = y[0]
    destination_y = y[1]
    deltaX = destination_x - origin_x
    deltaY = destination_y - origin_y
    degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp
    return degrees_final


def generate_navigation(path, target_lane):
    '''
    given the target lane
    firstly, compute its neighbour lane
    secondly, compute corresponding navigation goals for these two lanes
    '''
    
    # -----------------------------------------
    # spawn current location and random generate destination for our ego car
    # -----------------------------------------
    wayPoints = np.load(path + 'interaction/setting/wayPoints.npy')
    mergelane = np.load(path + 'interaction/setting/mergelane.npy')
    lane_cost = np.load(path + 'interaction/setting/lane_cost.npy')
    
    # -----------------------------------------
    # compute source and destination locations
    # -----------------------------------------
    for [left_lane, right_lane] in mergelane:
        if int(right_lane) == target_lane:
            print('left_lane:', left_lane)
            for i in range(0, len(wayPoints)):
                if wayPoints[i][5] == right_lane:
                    lane_length = lane_cost[int(right_lane) - 1]
                    front_waypoint_right = wayPoints[i + round(int(lane_length) / 3)]
                    back_waypoint_right = wayPoints[i + lane_length - 1]
                    break

            for i in range(0, len(wayPoints)):
                if wayPoints[i][5] == left_lane:
                    lane_length = lane_cost[int(left_lane) - 1]
                    front_waypoint_left = wayPoints[i + round(int(lane_length) / 3)]
                    back_waypoint_left = wayPoints[i + lane_length - 1]
                    break

            # option 1
            source_state1 = [back_waypoint_right[2], back_waypoint_right[3], back_waypoint_right[4]]
            dest_state1 = [front_waypoint_left[2], front_waypoint_left[3], front_waypoint_left[4]]
            x_option1 = [source_state1[1], source_state1[0]]
            y_option1 = [dest_state1[1], dest_state1[0]]
            direction1 = compute_direction(x_option1, y_option1)

            # option 2
            source_state2 = [front_waypoint_right[2], front_waypoint_right[3], front_waypoint_right[4]]
            dest_state2 = [back_waypoint_left[2], back_waypoint_left[3], back_waypoint_left[4]]
            x_option2 = [source_state2[1], source_state2[0]]
            y_option2 = [dest_state2[1], dest_state2[0]]
            direction2 = compute_direction(x_option2, y_option2)

            if abs(direction1 - source_state1[2]) < abs(direction2 - source_state1[2]):
                source_state = source_state1
                dest_state = dest_state1
            else:
                source_state = source_state2
                dest_state = dest_state2

    return source_state, dest_state


def remove_item_from_list(target_list, target_item):
    '''
    given a list, and a target item
    remove this item from the list
    '''
    temp_target_list = []
    for item in target_list:
        if item not in [target_item]:
            temp_target_list.append(item)
    target_list = temp_target_list
    return target_list


def remove_redundant_from_list(target_list):
    '''
    given a list
    remove redundant item from the list
    '''
    temp_target_list = []
    for item in target_list:
        if item not in temp_target_list:
            temp_target_list.append(item)
    target_list = temp_target_list
    return target_list


def unique_list(target_list):  
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for item in target_list:
        # check if exists in unique_list or not
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

def sort_item_in_list(target_list):
    # input: [[37, 0], [47, 4], [38, 1], [83, 2], [139, 3]]
    # output: [[37, 0], [38, 1], [83, 2], [139, 3], [47, 4]]
    
    # extract indexs and sort them  
    indexs = []
    for item in target_list:
        indexs.append(item[1])
    indexs = sorted(indexs)
    # print('sorted indexs:{}'.format(indexs))
    # select item according to index
    temp_target_list = []
    for itemX in indexs:
        for itemY in target_list:
            if itemX == itemY[1]:
                temp_target_list.append(itemY)
    return temp_target_list


def compatible(general_taskplan, visited):
    # general_taskplan = [['gas', 'school', 'grocery'],
                        # ['gas', 'grocery', 'school'],
                        # ['school', 'gas', 'grocery'],
                        # ['school', 'grocery', 'gas'],
                        # ['grocery', 'school', 'gas'],
                        # ['grocery', 'gas', 'school']]
    # visited = ['gas']
    # feasible_general_taskplan = [['gas', 'school', 'grocery', 'home'],
                                # ['gas', 'grocery', 'school', 'home']]
    if len(visited) == 0:
        # add 'init'
        temp_general_taskplan = []
        for item in general_taskplan:
            temp_general_taskplan.append(['init', item[0], item[1], item[2], 'home'])
        return temp_general_taskplan
    else:
        temp_feasible_general_taskplan = []
        if len(visited) == 1:
            for item in general_taskplan:
                if visited[0] == item[0]:
                    temp_feasible_general_taskplan.append([item[0], item[1], item[2], 'home'])
        if len(visited) == 2:
            for item in general_taskplan:
                if visited[0] == item[0] and visited[1] == item[1]:
                    temp_feasible_general_taskplan.append([item[1], item[2], 'home'])
        if len(visited) == 3:
            for item in general_taskplan:
                if visited[0] == item[0] and visited[1] == item[1] and visited[2] == item[2]:
                    temp_feasible_general_taskplan.append([item[2], 'home'])
        
        if len(temp_feasible_general_taskplan) <= 0:
            print('Error: no feasible general_taskplan')
        else:
            return temp_feasible_general_taskplan


def first_poi_2_lane(first_poi, visited_milestones, init_lane, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane, school_lane):
    if first_poi == 'init':
        return init_lane
    elif first_poi == 'gas':
        if gas1_lane in visited_milestones:
            return gas1_lane
        elif gas2_lane in visited_milestones:
            return gas2_lane
        else:
            print('Error: no lane id for first_poi (gas)')
    elif first_poi == 'grocery':
        if grocery1_lane in visited_milestones:
            return grocery1_lane
        elif grocery2_lane in visited_milestones:
            return grocery2_lane
        else:
            print('Error: no lane id for first_poi (grocery)')
    elif first_poi == 'school':
        return school_lane
    else:
        print('Error: no lane id for first_poi')


def mid_poi_2_lane(poi, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane, school_lane):
    # poi name -> poi lane
    if poi == 'gas':
        return [gas1_lane, gas2_lane]
    elif poi == 'grocery':
        return [grocery1_lane, grocery2_lane]
    elif poi == 'school':
        return [school_lane]
    else:
        print('Error: no lane id for mid poi')


def min_cost_2lanes(feasible_items):
    # output the optimal task plan with minmal cost between two lanes
    '''
    each item in feasible_items:
    item[0]: source_lane
    item[1]: dest_lane
    items[2]: task plan
    items[3]: plan_cost
    items[4]: plan_risk
    items[5]: flag_merge
    items[6]: poi_names
    items[7]: cost of each lane
    '''
    min_index = 0
    min_cost = 9999
    if len(feasible_items) > 0:
        for index in range(len(feasible_items)):
            if feasible_items[index][3] < min_cost:
                min_index = index
                min_cost = feasible_items[index][3]
        return feasible_items[min_index]
    else:
        print('Error: feasible_items is empty')


def min_cost_risk_2lanes(feasible_items, risk_alpha, lane_risk):
    # output the optimal task plan with minmal cost and risk between two lanes
    '''
    each item in feasible_items:
    item[0]: source_lane
    item[1]: dest_lane
    items[2]: task plan
    items[3]: plan_cost
    items[4]: plan_risk
    items[5]: flag_merge
    items[6]: poi_names
    items[7]: cost of each lane
    '''
    def risk_transform(risk_alpha, old):
        # new = math.log(old/100 + 1) * risk_alpha
        new = risk_alpha * old
        return new

    min_cost_risk = 9999
    min_index = 0
    min_cost = 0
    min_risk = 0
    for index in range(len(feasible_items)):
        item = feasible_items[index]
        # compute risk via traversing the lane_risk
        t_risk = 0
        for index_merge in range(len(item[5])):
            item_flagmerge = item[5]
            if item_flagmerge[index_merge] == 1: # if exists merge lane
                item_taskplan = item[2]
                risk = lane_risk[item_taskplan[index_merge] - 1]
                t_risk += risk_transform(risk_alpha, risk) # after transformation
        # compute t_risk + cost
        if item[3] + t_risk < min_cost_risk:
            min_cost_risk = item[3] + t_risk
            min_index = index
            min_cost = item[3]
            min_risk = t_risk
    return feasible_items[min_index], min_cost_risk, min_cost, min_risk


# difference
def min_cost_risk_2lanes_nocost(feasible_items, risk_alpha, lane_risk):
    # output the optimal task plan with minmal cost and risk between two lanes
    '''
    each item in feasible_items:
    item[0]: source_lane
    item[1]: dest_lane
    items[2]: task plan
    items[3]: plan_cost
    items[4]: plan_risk
    items[5]: flag_merge
    items[6]: poi_names
    items[7]: cost of each lane
    '''
    def risk_transform(risk_alpha, old):
        # new = math.log(old/100 + 1) * risk_alpha
        new = risk_alpha * old
        return new

    min_cost_risk = 9999
    min_index = 0
    min_cost = 0
    min_risk = 0
    for index in range(len(feasible_items)):
        item = feasible_items[index]
        # compute risk via traversing the lane_risk
        t_risk = 0
        for index_merge in range(len(item[5])):
            item_flagmerge = item[5]
            if item_flagmerge[index_merge] == 1: # if exists merge lane
                item_taskplan = item[2]
                risk = lane_risk[item_taskplan[index_merge] - 1]
                t_risk += risk_transform(risk_alpha, risk) # after transformation
        # compute t_risk + cost
        # difference
        if len(item[2]) * 80 + t_risk < min_cost_risk:
            min_cost_risk = item[3] + t_risk
            min_index = index
            # difference
            min_cost = len(item[2]) * 80
            # print('min_cost(nocost):{}'.format(min_cost))
            min_risk = t_risk
    return feasible_items[min_index], min_cost_risk, min_cost, min_risk



def preference_lanes(penalty, taskplan_lanes):
    '''
    each item in taskplan_lanes:
    item[0]: source_lane
    item[1]: dest_lane
    items[2]: task plan
    items[3]: plan_cost
    items[4]: plan_risk
    items[5]: flag_merge
    items[6]: poi_names
    items[7]: cost of each lane
    '''     
    def lanes_5(taskplan_lanes):
        taskplan_lanes_1_2 = taskplan_lanes[0]
        taskplan_lanes_2_3 = taskplan_lanes[1]
        taskplan_lanes_3_4 = taskplan_lanes[2]
        taskplan_lanes_4_5 = taskplan_lanes[3]
        POIs = [taskplan_lanes_1_2[6][0], taskplan_lanes_2_3[6][0], taskplan_lanes_3_4[6][0], taskplan_lanes_4_5[6][0], taskplan_lanes_4_5[6][1]]
        # print('poi_1:{} poi_2:{} poi_3:{} poi_4:{} poi_5:{}'.format(POIs[0], POIs[1], POIs[2], POIs[3], POIs[4]))
        return POIs
    
    def lanes_4(taskplan_lanes):
        taskplan_lanes_1_2 = taskplan_lanes[0]
        taskplan_lanes_2_3 = taskplan_lanes[1]
        taskplan_lanes_3_4 = taskplan_lanes[2]
        POIs = [taskplan_lanes_1_2[6][0], taskplan_lanes_2_3[6][0], taskplan_lanes_3_4[6][0], taskplan_lanes_3_4[6][1]]
        # print('poi_1:{} poi_2:{} poi_3:{} poi_4:{}'.format(POIs[0], POIs[1], POIs[2], POIs[3]))
        return POIs
    
    def lanes_3(taskplan_lanes):
        taskplan_lanes_1_2 = taskplan_lanes[0]
        taskplan_lanes_2_3 = taskplan_lanes[1]
        POIs = [taskplan_lanes_1_2[6][0], taskplan_lanes_2_3[6][0], taskplan_lanes_2_3[6][1]]
        # print('poi_1:{} poi_2:{} poi_3:{}'.format(POIs[0], POIs[1], POIs[2]))
        return POIs
    
    POIs = []
    # extract different POIs
    if len(taskplan_lanes) == 4:
        POIs = lanes_5(taskplan_lanes)
    elif len(taskplan_lanes) == 3:
        POIs = lanes_4(taskplan_lanes)
    elif len(taskplan_lanes) == 2:
        POIs = lanes_3(taskplan_lanes)
    else:
        print('Error: wrong length of taskplan_lanes')
     
    return compute_penalty(POIs, penalty)


def generate_remain_taskplan(visited_milestones, curr_lane, optm_plan, optm_flag_merge):
    index_curr_lane = optm_plan.index(curr_lane)
    remain_taskplan = optm_plan[index_curr_lane:]
    remain_flage_merge = optm_flag_merge[index_curr_lane:]
    return remain_taskplan, remain_flage_merge
    

def compute_dist(x, y, m, n):
    return math.sqrt((x - m) ** 2 + (y - n) ** 2)


def readX(path):  # X: coords_motionPlanner.txt
    f = open(path + 'coords_motionPlanner.txt', 'r')
    X_temp = []
    for line in f:
        X1 = [float(x) for x in line.split(',')]
        X_temp.append(X1)
    f.close()
    return X_temp


def update_risk(path, p_risk, curr_lane):
    lane_risk = np.load(path + 'lane_risk.npy')
    lane_risk[int(curr_lane)-1] = int(p_risk * 100)
    np.save(path + 'lane_risk.npy', lane_risk)
    return lane_risk


def update_visited(path, curr_lane):
    visited_milestones = list(np.load(path + 'visited_milestones.npy'))
    visited_milestones.append(int(curr_lane))
    np.save(path + 'visited_milestones.npy', visited_milestones)


def abstract_simulator_traf(jam):
    traffic = np.random.choice(['jam', 'no_jam'], p=[jam, 1.0 - jam])
    return traffic


def abstract_simulator_TP_FN(value_TP, value_FN):
    alpha = 1.0 / (value_TP + value_FN)
    value_TP = value_TP * alpha
    value_FN = 1.0 - value_TP
    result = np.random.choice(['TP', 'FN'], p=[value_TP, value_FN])
    return result


def abstract_simulator_TN_FP(value_TN, value_FP):
    alpha = 1.0 / (value_TN + value_FP)
    value_TN = value_TN * alpha
    value_FP = 1.0 - value_TN
    result = np.random.choice(['TN', 'FP'], p=[value_TN, value_FP])
    return result


def risk_FP(probs_FP, bins_FP):
    result = np.random.choice(bins_FP, p=probs_FP)
    return result


def risk_TP(probs_TP, bins_TP):
    result = np.random.choice(bins_TP, p=probs_TP)
    return result


def risk_FN(probs_FN, bins_FN):
    result = np.random.choice(bins_FN, p=probs_FN)
    return result

def risk_TN(probs_TN, bins_TN):
    result = np.random.choice(bins_TN, p=probs_TN)
    return result


def X_2_int(X):
    temp_X = []
    for item in X:
        temp_X.append(int(item))
    return temp_X


def X_2_float(X):
    temp_X = []
    for item in X:
        temp_X.append(float(item))
    return temp_X


def take_action(X):
    return X[1:]


def evaluate_plan_cost(path, taskplan):
    land_cost = np.load(path + 'lane_cost.npy')
    # cost part
    cost = 0
    for lane_id in taskplan:
        cost += land_cost[lane_id-1]
    return cost


def compute_penalty(POIs, penalty):
        total_penalty = 0
        # penalty1: visiting school before visiting gas1/gas2
        gasX_line = ''
        if 'gas1_lane' in POIs:
            gasX_line = 'gas1_lane'
        elif 'gas2_lane' in POIs:
            gasX_line = 'gas2_lane'

        if 'school_lane' in POIs and len(gasX_line) > 0:
            # school and gas are in POIs
            index_school = POIs.index('school_lane')
            index_gas = POIs.index(gasX_line)
            if index_school < index_gas:
                total_penalty += penalty
            else:
                total_penalty += 0
        else:
            total_penalty += 0
    
        # penalty2: visiting shool after grocery1/grocery2
        groceryX_line = ''
        if 'grocery1_lane' in POIs:
            groceryX_line = 'grocery1_lane'
        elif 'grocery2_lane' in POIs:
            groceryX_line = 'grocery2_lane'

        if 'school_lane' in POIs and len(groceryX_line) > 0:
            # school and grocery are in POIs
            index_school = POIs.index('school_lane')
            index_grocery = POIs.index(groceryX_line)
            if index_school > index_grocery:
                total_penalty += penalty
            else:
                total_penalty += 0
        else:
            total_penalty += 0

        return total_penalty


def evaluate_plan_pref(path, penalty, POIs_id):
    # translate id to name
    POIs_lane = np.load(path + 'POIs_lane.npy')
    init_lane = int(POIs_lane[0])
    home_lane = int(POIs_lane[1])
    school_lane = int(POIs_lane[2])
    gas1_lane = int(POIs_lane[3])
    gas2_lane = int(POIs_lane[4])
    grocery1_lane = int(POIs_lane[5])
    grocery2_lane = int(POIs_lane[6])
    
    POIs = []
    for id in POIs_id:
        if id == school_lane:
            POIs.append('school_lane')
        elif id == gas1_lane or id == gas2_lane:
            POIs.append('gas1_lane')
        elif id == grocery1_lane or id == grocery2_lane:
            POIs.append('grocery1_lane')
        else:
            print('Error: Unknown POIs')
    # print('POIs:{}'.format(POIs))
    
    return compute_penalty(POIs, penalty)


def TP_FN_FP_TN(TP_FN, precision, recall):
    TP = TP_FN * recall
    FN = TP_FN - TP
    FP = (1.0 / precision - 1) * TP
    TN = 1 - TP - FN - FP
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return TP, FN, FP, TN



def evaluate_utility(cost, pref, risk, constant):
    overall = - cost - pref - risk + constant
    return overall


def params_safety_estimator(model_name):
    if model_name == 'ANN_100':
        TP = 1387. / 3460.
        TN = 1564. / 3460. 
        FP = 252. / 3460. 
        FN = 257. / 3460.
    elif model_name == 'ANN_83':
        TP = 1418. / 3460.
        TN = 1518. / 3460. 
        FP = 221. / 3460. 
        FN = 303. / 3460.
    elif model_name == 'ANN_66':
        TP = 1393. / 3460.
        TN = 1529. / 3460. 
        FP = 246. / 3460. 
        FN = 292. / 3460.
    elif model_name == 'ANN_50':
        TP = 1350. / 3460.
        TN = 1560. / 3460. 
        FP = 289. / 3460. 
        FN = 261. / 3460.
    elif model_name == 'ANN_33':
        TP = 1346. / 3460.
        TN = 1550. / 3460. 
        FP = 293. / 3460. 
        FN = 271. / 3460.
    elif model_name == 'ANN_16':
        TP = 1311. / 3460.
        TN = 1496. / 3460. 
        FP = 328. / 3460. 
        FN = 325. / 3460.
    elif model_name == 'SVM_100':
        TP = 1321. / 3460.
        TN = 1469. / 3460. 
        FP = 318. / 3460. 
        FN = 352. / 3460.
    elif model_name == 'SVM_83':
        TP = 1320. / 3460.
        TN = 1463. / 3460. 
        FP = 319. / 3460. 
        FN = 358. / 3460.
    elif model_name == 'SVM_66':
        TP = 1309. / 3460.
        TN = 1459. / 3460. 
        FP = 330. / 3460. 
        FN = 362. / 3460.
    elif model_name == 'SVM_50':
        TP = 1302. / 3460.
        TN = 1423. / 3460. 
        FP = 337. / 3460. 
        FN = 398. / 3460.
    elif model_name == 'SVM_33':
        TP = 1283. / 3460.
        TN = 1397. / 3460. 
        FP = 356. / 3460. 
        FN = 424. / 3460.
    elif model_name == 'SVM_16':
        TP = 1270. / 3460.
        TN = 1361. / 3460. 
        FP = 369. / 3460. 
        FN = 460. / 3460.
    else:
        print('Error: unknown safety estimator')
    return TP, TN, FP, FN


def record_parameters(path, time, params_sim, params_optm, params_eva):
    fidout = open(path + time + '_parameters.txt', 'a')
    fidout.write('params_sim:{}\n params_optm:{}\n params_eva:{}\n'.format(params_sim, params_optm, params_eva))


def reset(path1, path3):
    # -----------------------------------------
    # load POIs
    # -----------------------------------------
    POIs_lane = np.load(path3 + 'POIs_lane.npy')
    init_lane = int(POIs_lane[0])

    np.save(path1 + 'current_lane.npy', init_lane)
    np.save(path1 + 'visited_milestones.npy', [])
    np.save(path1 + 'lane_risk.npy', [0]*156)
    
    # print('current state is reset!')


def Nbatch_2_1batch(Nbatch):
    # -----------------------------------------
    # N batchs - > 1 batch
    # -----------------------------------------
    # test
    temp_batch = []
    for index in range(len(Nbatch[0])):
        temp_value = 0
        for batch in Nbatch:
            temp_value += batch[index]
        temp_batch.append(temp_value/len(Nbatch))
    return temp_batch


def get_filename(path):
    filename_list = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        filename_list.extend(filenames)
    return filename_list


def Nbatch_transform(Nbatch):
    # 200 batch, each batch has 100 trials
    # -> 2000 batch, each batch has 10 traisl
    data = []
    # read all data
    new_Nbatch = []
    # print('-'*30)
    # print('old Nbatch: {}'.format(Nbatch))
    for line in Nbatch:
        # print('-'*30)
        # print('line: {}'.format(line))
        # M = int(len(line) / 10)
        # temp = [np.mean(line[0:M]), np.mean(line[M:2*M]), np.mean(line[2*M:3*M]), np.mean(line[3*M:4*M]), np.mean(line[3*M:4*M]), np.mean(line[4*M:5*M]), np.mean(line[5*M:6*M]), np.mean(line[6*M:7*M]), np.mean(line[7*M:8*M]), np.mean(line[8*M:9*M]), np.mean(line[9*M:10*M])]
        M = int(len(line) / 2)
        temp = [np.mean(line[0:M]), np.mean(line[M:2*M])]
        # print('-'*30)
        # print('temp: {}'.format(temp))
        new_Nbatch.append(temp)
        # print('-'*30)
        # print('new_Nbatch: {}'.format(new_Nbatch))
    return new_Nbatch


def read_result(folder_path, threshold_cost, penalty_cost, penalty_risk, constant):
    data_method_overall1 = []
    data_method_overall2 = []
    data_method_cost = []
    data_method_pref = []
    data_method_brisk = []
    data_method_risk = []
    data_method_avoid = []
    data_method_collision = []
    filename_list = get_filename(folder_path)
    for filename in filename_list:
        if '.npy' in filename:
            temp_data = np.load(folder_path + filename)
            # data of one batch
            batch_data_method_overall1 = []
            batch_data_method_overall2 = []
            batch_data_method_cost = []
            batch_data_method_pref = []
            batch_data_method_brisk = []
            batch_data_method_collision = []
            batch_data_method_risk = []
            batch_data_method_avoid = []
            for item in temp_data:
                cost = item[0]
                if cost > threshold_cost:
                    cost = cost + penalty_cost
                pref = item[1]
                brisk = item[2]
                collision = item[3]
                risk = item[4]
                avoid = item[5]
                # option 1: if has risk, then give a big penalty
                overall1 = evaluate_utility(cost, pref, brisk*penalty_risk, constant)
                # option 2: if has N risk, then give a big penalty * N
                overall2 = evaluate_utility(cost, pref, risk*penalty_risk, constant)
                batch_data_method_overall1.append(overall1)
                batch_data_method_overall2.append(overall2)
                batch_data_method_cost.append(cost)
                batch_data_method_pref.append(pref)
                batch_data_method_brisk.append(brisk)
                batch_data_method_collision.append(collision)
                batch_data_method_risk.append(risk)
                batch_data_method_avoid.append(avoid)
            data_method_overall1.append(batch_data_method_overall1)
            data_method_overall2.append(batch_data_method_overall2)
            data_method_cost.append(batch_data_method_cost)
            data_method_pref.append(batch_data_method_pref)
            data_method_brisk.append(batch_data_method_brisk)
            data_method_collision.append(batch_data_method_collision)
            data_method_risk.append(batch_data_method_risk)
            data_method_avoid.append(batch_data_method_avoid)
    data_method_overall1 = Nbatch_2_1batch(data_method_overall1)
    data_method_overall2 = Nbatch_2_1batch(data_method_overall2)
    data_method_cost = Nbatch_2_1batch(data_method_cost)
    data_method_pref = Nbatch_2_1batch(data_method_pref)
    data_method_brisk = Nbatch_2_1batch(data_method_brisk)
    data_method_collision = Nbatch_2_1batch(data_method_collision)
    data_method_risk = Nbatch_2_1batch(data_method_risk)
    data_method_avoid= Nbatch_2_1batch(data_method_avoid)
    
    # return data_method_overall1, data_method_overall2, data_method_cost, data_method_pref, data_method_brisk, data_method_risk
    return data_method_overall1, data_method_overall2, data_method_cost, data_method_pref, data_method_brisk, data_method_collision, data_method_risk, data_method_avoid


def transform_risk(case, old):
    threshold_P = 0.7
    threshold_N = 0.3
    if case == 'TP' or case == 'FP':
        if old <= threshold_P:
            new = threshold_P
        elif old >= threshold_P:
            new = old
        else:
            print('Error: no P case')
    elif case == 'TN' or case == 'FN':
        if old >= threshold_N:
            new = threshold_N
        elif old <= threshold_N:
            new = old
        else:
            print('Error: no N case')
    return new