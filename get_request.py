#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# This code is generate request for each trial!
# -----------------------------------------

from __future__ import print_function
import os
import getpass
import random
import numpy as np
import utils

# -----------------------------------------
# definite paths
# -----------------------------------------
path = utils.root_path() + 'interaction/setting/'

# -----------------------------------------
# set initial location and destination
# -----------------------------------------
init_lane = 37 # initial location of the ego car
home_lane = 47 # the destination

# -----------------------------------------
# set semantic locations
# -----------------------------------------
if not os.path.exists(path + 'POIs_candidate.npy'):
    if not os.path.exists(path):
        os.mkdir(path)
    
    candidate_lanes = [103, 106,   1,   4,  44,  41,  73,  76,  80,  77,  49,  52,  69,
                        72,  81,  84,  21,  24,  40, 140, 139,   5,   8, 135, 138, 107,
                       110,  53,  56, 114, 111,  25,  28, 131, 124,   9,  12,  17,  20,
                       145, 148,  57,  60, 141, 144,  29,  32, 153, 156,  33,  36, 127,
                       130, 123, 126,  13,  16, 119, 122,  61,  64, 115, 118,  46,  45,
                       149, 152,  48,  65,  68, 102,  97,  91,  96,  85,  90]

    # -----------------------------------------
    # remove init and home lanes
    # -----------------------------------------
    candidate_lanes = utils.remove_item_from_list(candidate_lanes, init_lane)
    candidate_lanes = utils.remove_item_from_list(candidate_lanes, home_lane)

    # -----------------------------------------
    # remove redundant lanes
    # -----------------------------------------
    candidate_lanes = utils.remove_redundant_from_list(candidate_lanes)
    
    np.save(path + 'POIs_candidate.npy', candidate_lanes)  # save candidate lanes

else:
    candidate_lanes = np.load(path + 'POIs_candidate.npy')  # load candidate lanes

# -----------------------------------------
# select six candidate lanes for POIs in our task
# -----------------------------------------
selected_lanes = random.sample(list(candidate_lanes), 6)

# option 1: randomly select POIs
school_lane = selected_lanes[0]
gas1_lane = selected_lanes[1]
gas2_lane = selected_lanes[2]
grocery1_lane = selected_lanes[4]
grocery2_lane = selected_lanes[5]

# option 2: manually set POIs
school_lane = 17
gas1_lane = 107
gas2_lane = 60
grocery1_lane = 72
grocery2_lane = 138

print('-'*30)
print('after generating request,\n init: {}\n home: {}\n school: {}\n gas1: {}\n gas2: {}\n grocery1: {}\n grocery2: {}'.format(init_lane, home_lane, school_lane, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane))
print('-'*30)

# save these selected POIs
np.save(path + 'POIs_lane.npy', [init_lane, home_lane, school_lane, gas1_lane, gas2_lane, grocery1_lane, grocery2_lane])