# -----------------------------------------
# This code is run simulation (abstract)
# -----------------------------------------

from __future__ import print_function
import os
import random
import time
import numpy as np
import getpass

# -----------------------------------------
# customized functions
# -----------------------------------------
import utils
from get_taskplan import get_taskplan
from get_motionplan import get_motionplan

np.random.seed(42)

name = 'KNN' # model name
# -----------------------------------------
# parameters
# -----------------------------------------
# abstract simulator
params_sim = {
    'jam': 0.08, # traffic jam
    'safety_estimator': name,
    'probs_FP': np.load(utils.root_path() + 'interaction/setting/FP_probs_' + name + '.npy'), 
    'bins_FP': np.load(utils.root_path() + 'interaction/setting/FP_bins_' + name + '.npy'),
    'probs_TP': np.load(utils.root_path() + 'interaction/setting/TP_probs_' + name + '.npy'), 
    'bins_TP': np.load(utils.root_path() + 'interaction/setting/TP_bins_' + name + '.npy'),
    'probs_FN': np.load(utils.root_path() + 'interaction/setting/FN_probs_' + name + '.npy'), 
    'bins_FN': np.load(utils.root_path() + 'interaction/setting/FN_bins_' + name + '.npy'),
    'probs_TN': np.load(utils.root_path() + 'interaction/setting/TN_probs_' + name + '.npy'), 
    'bins_TN': np.load(utils.root_path() + 'interaction/setting/TN_bins_' + name + '.npy'),
}
params_sim['bins_FP'] = params_sim['bins_FP'][1:]
params_sim['bins_TP'] = params_sim['bins_TP'][1:]
params_sim['bins_FN'] = params_sim['bins_FN'][1:]
params_sim['bins_TN'] = params_sim['bins_TN'][1:]

# full simulation
params_real = {
    'real': False,
}
# optimization
params_optm = {
    'penalty': 300, # violate user preference
    'risk_alpha': 500,
}
# evaluation
params_eva = {
    'batch': 100,
    'epoch': 100,
}
# debugging
params_debug = {
    'print': False, # if debugging is enabled
    'print_freq': 50,
    'path_batch': utils.root_path() + 'test_' + name + '_' + str(int(params_sim['jam'] * 100)) + '/',
}
if not os.path.exists(params_debug['path_batch']):
    os.mkdir(params_debug['path_batch'])
# record parameters
init_time = int(time.time())
utils.record_parameters(params_debug['path_batch'], str(init_time), params_sim, params_optm, params_eva)

# -----------------------------------------
# compute TP, FN, FP, TN of safety esimator
# -----------------------------------------
TP, TN, FP, FN = utils.params_safety_estimator(params_sim['safety_estimator'])
if params_debug['print']:
    print('-'*30)
    print('TP: {} TN: {} FP: {} FN: {}\n'.format(TP, TN, FP, FN))


# -----------------------------------------
# definite paths
# -----------------------------------------
path1 = utils.root_path() + 'interaction/'
path2 = utils.root_path() + 'task-level/'
path3 = utils.root_path() + 'interaction/setting/'

# -----------------------------------------
# running
# -----------------------------------------
for batch in range(params_eva['batch']):
    batch += 1
    batch_result = []    
    for epoch in range(params_eva['epoch']):
        if epoch % params_debug['print_freq'] == 0:
            print('batch:{} epoch:{}'.format(batch, epoch))

        # initially task and motion planner
        utils.reset(path1, path3)
        get_taskplan(utils.root_path(), params_optm['risk_alpha'], params_optm['penalty'])
        get_motionplan(utils.root_path())

        # load POIs
        POIs_lane = np.load(path3 + 'POIs_lane.npy')
        init_lane = int(POIs_lane[0])
        home_lane = int(POIs_lane[1])

        X_site = list(np.load(path1 + 'site.npy')) # load trajctory: site
        curr_lane = int(X_site[0][2])
        flag_merge = int(X_site[0][1])  # 1: merge; 0: not merge
        if params_debug['print']:
            print('current lane:{}  flage of merge:{}'.format(curr_lane, flag_merge))

        # record running result
        visited_lanes = []
        num_collision = 0 # collisions in simulation process
        has_collision = 0 # collisions that our car has
        
        # -----------------------------------------
        # planning time
        # -----------------------------------------
        while curr_lane != home_lane:
            if flag_merge == 0:
                if params_debug['print']:
                    print('take action!')
                X_site = utils.take_action(X_site)
                curr_lane = int(X_site[0][2])
                flag_merge = int(X_site[0][1])
                np.save(path1 + 'current_lane.npy', curr_lane)
                if params_debug['print']:
                    print('current lane:{}  flag of merge:{}'.format(curr_lane, flag_merge))
                visited_lanes.append(curr_lane)

                # update visited milestone
                milestones = np.load(path1 + 'optm_milestone.npy')
                if curr_lane in milestones and curr_lane not in [init_lane, home_lane]:
                    utils.update_visited(path1, curr_lane)
            else:
                # -----------------------------------------
                # simulate risk situation
                # -----------------------------------------
                traffic_condition = utils.abstract_simulator_traf(params_sim['jam']) # jam or no_jam
                if traffic_condition == 'jam':
                    num_collision += 1
                    predict_result = utils.abstract_simulator_TP_FN(TP, FN)
                    if predict_result == 'TP':
                        p_risk = utils.risk_TP(params_sim['probs_TP'], params_sim['bins_TP'])
                    elif predict_result == 'FN':
                        num_collision += 1
                        p_risk = utils.risk_FN(params_sim['probs_FN'], params_sim['bins_FN'])
                    else:
                        print('Error: unknown result TP FN')
                elif traffic_condition == 'no_jam':
                    predict_result = utils.abstract_simulator_TN_FP(TN, FP)
                    if predict_result == 'TN':
                        p_risk = utils.risk_TN(params_sim['probs_TN'], params_sim['bins_TN'])
                    elif predict_result == 'FP':
                        p_risk = utils.risk_FP(params_sim['probs_FP'], params_sim['bins_FP'])
                    else:
                        print('Error: unknown result TN FP')
                else:
                    print('Error: unknown traffic condition')

                # if not params_debug['print']:
                #     print('-'*30)
                #     print('traffic condition: {} predict result:{} p_risk:{}\n'.format(traffic_condition, predict_result, p_risk))

                # -----------------------------------------
                # update risk and re-run task and motion planner
                # -----------------------------------------
                lane_risk = utils.update_risk(path1, p_risk, curr_lane)
                get_taskplan(utils.root_path(), params_optm['risk_alpha'], params_optm['penalty'])
                get_motionplan(utils.root_path())
                
                # load motion trajctory
                X_site = list(np.load(path1 + 'site.npy')) # load trajctory: site

                flag_merge = X_site[0][1]
                if flag_merge == 1: # no other plan is found
                    # when has a collision
                    if traffic_condition == 'jam':
                        has_collision += 1
                        if params_debug['print']:
                            print('-'*30)
                            print('result:{} not change lane -> a collison'.format(result))
                    else:
                        if params_debug['print']:
                            print('-'*30)
                            print('result:{} not change lane -\> no collison'.format(result))

                    if params_debug['print']:
                        print('force to take action!')

                    X_site = utils.take_action(X_site)
                    curr_lane = int(X_site[0][2])
                    flag_merge = int(X_site[0][1])
                    np.save(path1 + 'current_lane.npy', curr_lane)
                    if params_debug['print']:
                        print('current lane:{}  flag of merge:{}'.format(curr_lane, flag_merge))
                    visited_lanes.append(curr_lane)

                    # update visited milestone
                    milestones = np.load(path1 + 'optm_milestone.npy')
                    if curr_lane in milestones and curr_lane not in [init_lane, home_lane]:
                        utils.update_visited(path1, curr_lane)

        # -----------------------------------------
        # evaluation time
        # -----------------------------------------
        # cost
        final_cost = utils.evaluate_plan_cost(path3, visited_lanes)

        # preference
        visited_milestones = np.load(path1 + 'visited_milestones.npy')
        fina_pref = utils.evaluate_plan_pref(path3, params_optm['penalty'], visited_milestones)

        # risk
        if has_collision == 0:
            final_risk = 0
        else:
            final_risk = 1

        # save batch result: cost, preference, risk, number of avoiding risk
        batch_result.append([final_cost, fina_pref, final_risk, num_collision, has_collision, (num_collision - has_collision)])

        if params_debug['print']:
            print('-'*30)
            print('cost:{} preference:{} brisk:{} num_collision:{} has_collision:{} avoiding collision:{}'.format(final_cost, fina_pref, final_risk, num_collision, has_collision, (num_collision - has_collision)))

        if params_real['real']:
            fidout = open(path1 + 'real_result.txt', 'a')
            fidout.write('-'*30)
            fidout.write('visited_lanes:{}\n'.format(visited_lanes))
            fidout.write('cost:{} preference:{} brisk:{} num_collision:{} has_collision:{} avoiding collision:{}'.format(final_cost, fina_pref, final_risk, num_collision, has_collision, (num_collision - has_collision)))

    filename = str(init_time) + '_our_batch_' + str(batch) + '.npy'
    np.save(params_debug['path_batch'] + filename, batch_result)

    print('computing time cost for a batch (s):{}'.format(time.time() - init_time))