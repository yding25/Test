#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot curve!
# -----------------------------------------

from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import os
import sys
sys.path.append('/home/yan/CARLA_0.9.10.1/PythonAPI/Test')
import utils
from mpl_toolkits import mplot3d
from matplotlib.pyplot import MultipleLocator


# -----------------------------------------
# styles
# -----------------------------------------
params = {
    # color
    # 'color': ['#C82423', '#2878B5', '#9AC9DB', '#F3D266', '#FF8884'],
    'color': ['#9AC9DB', '#F8AC8C', '#2878B5'],

    'ecolor': ['black'],
    
    'marker': ['d', '^', 'o', 's', 's', '>'], # refer to https://matplotlib.org/stable/api/markers_api.html
    'marker_size': 22,
    
    'bar_width': 0.3,
    'bar_interval': 0.2,
    
    'error_capsize': 5,
    
    'xaxis_fontsize': 12,
    'xaxis_degree': 0,
    'xaxis_fontweight': 'normal',
    'xlabel_fontsize': 18,
    'xlabel_fontweight': 'bold',
    'xlim_min': 0,
    'xlim_max': 10,

    'yaxis_fontsize': 12,
    'yaxis_degree': 90,
    'yaxis_fontweight': 'normal',
    'ylabel_fontsize': 18,
    'ylabel_fontweight': 'bold',
    'ylim_min': 0,
    'ylim_max': 92,

    'legend_fontsize': 14,
    'legend_location': ['best', 'upper right'],
    
    'figure_width': 12,
    'figure_height': 8,
    
    'grid': True,

    'fill_transparent': 0.5,

    'line_width': 3.,
    'line_style': ['-', '--', ':', '-.'],

    'size': np.array([300, 300, 300, 300, 300]),

    'hatch': ['\\\\', 'o', '//', '.', '-', 'x', 'o', 'O', '+', '*'],
    
    'fig_name': 'fig_utility2',
    'fig_svg': 'svg',
    'fig_png': 'png',
    'fig_jpeg': 'jpeg',
    'fig_pdf': 'pdf',
    'fig_transparent': False,
}
params_eva = {
    'threshold_cost': 5000, # if the task is completed within time threshold
    'penalty_cost': 0, # cost penalty
    'penalty_risk': 15000, # risk penalty
    'constant': 0,
}
params_debug = {
    'print': False
}


# -----------------------------------------
# data processing: Our (ANN_100, 8)
# -----------------------------------------
path_our_ANN_100_8 = utils.root_path() + 'results/8/test_ANN_100_8_nonew/'
data_our_ANN_100_8_overall1, data_our_ANN_100_8_overall2, data_our_ANN_100_8_cost, data_our_ANN_100_8_pref, data_our_ANN_100_8_brisk, data_our_ANN_100_8_collision, data_our_ANN_100_8_risk, data_our_ANN_100_8_avoid = utils.read_result(path_our_ANN_100_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])

if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_100_8_overall1:{}'.format((data_our_ANN_100_8_overall1)))
print('-'*30)
print('data_our_ANN_100_8_overall1:{}'.format(np.mean(data_our_ANN_100_8_overall1)))
print('data_our_ANN_100_8_cost:{}'.format(np.mean(data_our_ANN_100_8_cost)))
print('data_our_ANN_100_8_pref:{}'.format(np.mean(data_our_ANN_100_8_pref)))
print('data_our_ANN_100_8_brisk:{}'.format(np.mean(data_our_ANN_100_8_brisk)))
print('data_our_ANN_100_8_collision:{}'.format(np.mean(data_our_ANN_100_8_collision)))
print('data_our_ANN_100_8_risk:{}'.format(np.mean(data_our_ANN_100_8_risk)))
print('data_our_ANN_100_8_avoid:{}'.format(np.mean(data_our_ANN_100_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_100, 5)
# -----------------------------------------
path_our_ANN_100_5 = utils.root_path() + 'results/5/test_ANN_100_5_nonew/'
data_our_ANN_100_5_overall1, data_our_ANN_100_5_overall2, data_our_ANN_100_5_cost, data_our_ANN_100_5_pref, data_our_ANN_100_5_brisk, data_our_ANN_100_5_collision, data_our_ANN_100_5_risk, data_our_ANN_100_5_avoid = utils.read_result(path_our_ANN_100_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])

if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_100_5_overall1:{}'.format((data_our_ANN_100_5_overall1)))
print('-'*30)
print('data_our_ANN_100_5_overall1:{}'.format(np.mean(data_our_ANN_100_5_overall1)))
print('data_our_ANN_100_5_cost:{}'.format(np.mean(data_our_ANN_100_5_cost)))
print('data_our_ANN_100_5_pref:{}'.format(np.mean(data_our_ANN_100_5_pref)))
print('data_our_ANN_100_5_brisk:{}'.format(np.mean(data_our_ANN_100_5_brisk)))
print('data_our_ANN_100_5_collision:{}'.format(np.mean(data_our_ANN_100_5_collision)))
print('data_our_ANN_100_5_risk:{}'.format(np.mean(data_our_ANN_100_5_risk)))
print('data_our_ANN_100_5_avoid:{}'.format(np.mean(data_our_ANN_100_5_avoid)))

# -----------------------------------------
# data processing: Our (ANN_83, 8)
# -----------------------------------------
path_our_ANN_83_8 = utils.root_path() + 'results/8/test_ANN_83_8_nonew/'
data_our_ANN_83_8_overall1, data_our_ANN_83_8_overall2, data_our_ANN_83_8_cost, data_our_ANN_83_8_pref, data_our_ANN_83_8_brisk, data_our_ANN_83_8_collision, data_our_ANN_83_8_risk, data_our_ANN_83_8_avoid = utils.read_result(path_our_ANN_83_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_83_8_overall1:{}'.format((data_our_ANN_83_8_overall1)))
print('-'*30)
print('data_our_ANN_83_8_overall1:{}'.format(np.mean(data_our_ANN_83_8_overall1)))
print('data_our_ANN_83_8_cost:{}'.format(np.mean(data_our_ANN_83_8_cost)))
print('data_our_ANN_83_8_pref:{}'.format(np.mean(data_our_ANN_83_8_pref)))
print('data_our_ANN_83_8_brisk:{}'.format(np.mean(data_our_ANN_83_8_brisk)))
print('data_our_ANN_83_8_collision:{}'.format(np.mean(data_our_ANN_83_8_collision)))
print('data_our_ANN_83_8_risk:{}'.format(np.mean(data_our_ANN_83_8_risk)))
print('data_our_ANN_83_8_avoid:{}'.format(np.mean(data_our_ANN_83_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_83, 5)
# -----------------------------------------
path_our_ANN_83_5 = utils.root_path() + 'results/5/test_ANN_83_5_nonew/'
data_our_ANN_83_5_overall1, data_our_ANN_83_5_overall2, data_our_ANN_83_5_cost, data_our_ANN_83_5_pref, data_our_ANN_83_5_brisk, data_our_ANN_83_5_collision, data_our_ANN_83_5_risk, data_our_ANN_83_5_avoid = utils.read_result(path_our_ANN_83_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_83_5_overall1:{}'.format((data_our_ANN_83_5_overall1)))
print('-'*30)
print('data_our_ANN_83_5_overall1:{}'.format(np.mean(data_our_ANN_83_5_overall1)))
print('data_our_ANN_83_5_cost:{}'.format(np.mean(data_our_ANN_83_5_cost)))
print('data_our_ANN_83_5_pref:{}'.format(np.mean(data_our_ANN_83_5_pref)))
print('data_our_ANN_83_5_brisk:{}'.format(np.mean(data_our_ANN_83_5_brisk)))
print('data_our_ANN_83_5_collision:{}'.format(np.mean(data_our_ANN_83_5_collision)))
print('data_our_ANN_83_5_risk:{}'.format(np.mean(data_our_ANN_83_5_risk)))
print('data_our_ANN_83_5_avoid:{}'.format(np.mean(data_our_ANN_83_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_66, 8)
# -----------------------------------------
path_our_ANN_66_8 = utils.root_path() + 'results/8/test_ANN_66_8_nonew/'
data_our_ANN_66_8_overall1, data_our_ANN_66_8_overall2, data_our_ANN_66_8_cost, data_our_ANN_66_8_pref, data_our_ANN_66_8_brisk, data_our_ANN_66_8_collision, data_our_ANN_66_8_risk, data_our_ANN_66_8_avoid = utils.read_result(path_our_ANN_66_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_66_8_overall1:{}'.format((data_our_ANN_66_8_overall1)))
print('-'*30)
print('data_our_ANN_66_8_overall1:{}'.format(np.mean(data_our_ANN_66_8_overall1)))
print('data_our_ANN_66_8_cost:{}'.format(np.mean(data_our_ANN_66_8_cost)))
print('data_our_ANN_66_8_pref:{}'.format(np.mean(data_our_ANN_66_8_pref)))
print('data_our_ANN_66_8_brisk:{}'.format(np.mean(data_our_ANN_66_8_brisk)))
print('data_our_ANN_66_8_collision:{}'.format(np.mean(data_our_ANN_66_8_collision)))
print('data_our_ANN_66_8_risk:{}'.format(np.mean(data_our_ANN_66_8_risk)))
print('data_our_ANN_66_8_avoid:{}'.format(np.mean(data_our_ANN_66_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_66, 5)
# -----------------------------------------
path_our_ANN_66_5 = utils.root_path() + 'results/5/test_ANN_66_5_nonew/'
data_our_ANN_66_5_overall1, data_our_ANN_66_5_overall2, data_our_ANN_66_5_cost, data_our_ANN_66_5_pref, data_our_ANN_66_5_brisk, data_our_ANN_66_5_collision, data_our_ANN_66_5_risk, data_our_ANN_66_5_avoid = utils.read_result(path_our_ANN_66_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_66_5_overall1:{}'.format((data_our_ANN_66_5_overall1)))
print('-'*30)
print('data_our_ANN_66_5_overall1:{}'.format(np.mean(data_our_ANN_66_5_overall1)))
print('data_our_ANN_66_5_cost:{}'.format(np.mean(data_our_ANN_66_5_cost)))
print('data_our_ANN_66_5_pref:{}'.format(np.mean(data_our_ANN_66_5_pref)))
print('data_our_ANN_66_5_brisk:{}'.format(np.mean(data_our_ANN_66_5_brisk)))
print('data_our_ANN_66_5_collision:{}'.format(np.mean(data_our_ANN_66_5_collision)))
print('data_our_ANN_66_5_risk:{}'.format(np.mean(data_our_ANN_66_5_risk)))
print('data_our_ANN_66_5_avoid:{}'.format(np.mean(data_our_ANN_66_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_50, 8)
# -----------------------------------------
path_our_ANN_50_8 = utils.root_path() + 'results/8/test_ANN_50_8_nonew/'
data_our_ANN_50_8_overall1, data_our_ANN_50_8_overall2, data_our_ANN_50_8_cost, data_our_ANN_50_8_pref, data_our_ANN_50_8_brisk, data_our_ANN_50_8_collision, data_our_ANN_50_8_risk, data_our_ANN_50_8_avoid = utils.read_result(path_our_ANN_50_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_50_8_overall1:{}'.format((data_our_ANN_50_8_overall1)))
print('-'*30)
print('data_our_ANN_50_8_overall1:{}'.format(np.mean(data_our_ANN_50_8_overall1)))
print('data_our_ANN_50_8_cost:{}'.format(np.mean(data_our_ANN_50_8_cost)))
print('data_our_ANN_50_8_pref:{}'.format(np.mean(data_our_ANN_50_8_pref)))
print('data_our_ANN_50_8_brisk:{}'.format(np.mean(data_our_ANN_50_8_brisk)))
print('data_our_ANN_50_8_collision:{}'.format(np.mean(data_our_ANN_50_8_collision)))
print('data_our_ANN_50_8_risk:{}'.format(np.mean(data_our_ANN_50_8_risk)))
print('data_our_ANN_50_8_avoid:{}'.format(np.mean(data_our_ANN_50_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_50, 5)
# -----------------------------------------
path_our_ANN_50_5 = utils.root_path() + 'results/5/test_ANN_50_5_nonew/'
data_our_ANN_50_5_overall1, data_our_ANN_50_5_overall2, data_our_ANN_50_5_cost, data_our_ANN_50_5_pref, data_our_ANN_50_5_brisk, data_our_ANN_50_5_collision, data_our_ANN_50_5_risk, data_our_ANN_50_5_avoid = utils.read_result(path_our_ANN_50_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_50_5_overall1:{}'.format((data_our_ANN_50_5_overall1)))
print('-'*30)
print('data_our_ANN_50_5_overall1:{}'.format(np.mean(data_our_ANN_50_5_overall1)))
print('data_our_ANN_50_5_cost:{}'.format(np.mean(data_our_ANN_50_5_cost)))
print('data_our_ANN_50_5_pref:{}'.format(np.mean(data_our_ANN_50_5_pref)))
print('data_our_ANN_50_5_brisk:{}'.format(np.mean(data_our_ANN_50_5_brisk)))
print('data_our_ANN_50_5_collision:{}'.format(np.mean(data_our_ANN_50_5_collision)))
print('data_our_ANN_50_5_risk:{}'.format(np.mean(data_our_ANN_50_5_risk)))
print('data_our_ANN_50_5_avoid:{}'.format(np.mean(data_our_ANN_50_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_33, 8)
# -----------------------------------------
path_our_ANN_33_8 = utils.root_path() + 'results/8/test_ANN_33_8_nonew/'
data_our_ANN_33_8_overall1, data_our_ANN_33_8_overall2, data_our_ANN_33_8_cost, data_our_ANN_33_8_pref, data_our_ANN_33_8_brisk, data_our_ANN_33_8_collision, data_our_ANN_33_8_risk, data_our_ANN_33_8_avoid = utils.read_result(path_our_ANN_33_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_33_8_overall1:{}'.format((data_our_ANN_33_8_overall1)))
print('-'*30)
print('data_our_ANN_33_8_overall1:{}'.format(np.mean(data_our_ANN_33_8_overall1)))
print('data_our_ANN_33_8_cost:{}'.format(np.mean(data_our_ANN_33_8_cost)))
print('data_our_ANN_33_8_pref:{}'.format(np.mean(data_our_ANN_33_8_pref)))
print('data_our_ANN_33_8_brisk:{}'.format(np.mean(data_our_ANN_33_8_brisk)))
print('data_our_ANN_33_8_collision:{}'.format(np.mean(data_our_ANN_33_8_collision)))
print('data_our_ANN_33_8_risk:{}'.format(np.mean(data_our_ANN_33_8_risk)))
print('data_our_ANN_33_8_avoid:{}'.format(np.mean(data_our_ANN_33_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_33, 5)
# -----------------------------------------
path_our_ANN_33_5 = utils.root_path() + 'results/5/test_ANN_33_5_nonew/'
data_our_ANN_33_5_overall1, data_our_ANN_33_5_overall2, data_our_ANN_33_5_cost, data_our_ANN_33_5_pref, data_our_ANN_33_5_brisk, data_our_ANN_33_5_collision, data_our_ANN_33_5_risk, data_our_ANN_33_5_avoid = utils.read_result(path_our_ANN_33_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_33_5_overall1:{}'.format((data_our_ANN_33_5_overall1)))
print('-'*30)
print('data_our_ANN_33_5_overall1:{}'.format(np.mean(data_our_ANN_33_5_overall1)))
print('data_our_ANN_33_5_cost:{}'.format(np.mean(data_our_ANN_33_5_cost)))
print('data_our_ANN_33_5_pref:{}'.format(np.mean(data_our_ANN_33_5_pref)))
print('data_our_ANN_33_5_brisk:{}'.format(np.mean(data_our_ANN_33_5_brisk)))
print('data_our_ANN_33_5_collision:{}'.format(np.mean(data_our_ANN_33_5_collision)))
print('data_our_ANN_33_5_risk:{}'.format(np.mean(data_our_ANN_33_5_risk)))
print('data_our_ANN_33_5_avoid:{}'.format(np.mean(data_our_ANN_33_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_16, 8)
# -----------------------------------------
path_our_ANN_16_8 = utils.root_path() + 'results/8/test_ANN_16_8_nonew/'
data_our_ANN_16_8_overall1, data_our_ANN_16_8_overall2, data_our_ANN_16_8_cost, data_our_ANN_16_8_pref, data_our_ANN_16_8_brisk, data_our_ANN_16_8_collision, data_our_ANN_16_8_risk, data_our_ANN_16_8_avoid = utils.read_result(path_our_ANN_16_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_16_8_overall1:{}'.format((data_our_ANN_16_8_overall1)))
print('-'*30)
print('data_our_ANN_16_8_overall1:{}'.format(np.mean(data_our_ANN_16_8_overall1)))
print('data_our_ANN_16_8_cost:{}'.format(np.mean(data_our_ANN_16_8_cost)))
print('data_our_ANN_16_8_pref:{}'.format(np.mean(data_our_ANN_16_8_pref)))
print('data_our_ANN_16_8_brisk:{}'.format(np.mean(data_our_ANN_16_8_brisk)))
print('data_our_ANN_16_8_collision:{}'.format(np.mean(data_our_ANN_16_8_collision)))
print('data_our_ANN_16_8_risk:{}'.format(np.mean(data_our_ANN_16_8_risk)))
print('data_our_ANN_16_8_avoid:{}'.format(np.mean(data_our_ANN_16_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_16, 5)
# -----------------------------------------
path_our_ANN_16_5 = utils.root_path() + 'results/5/test_ANN_16_5_nonew/'
data_our_ANN_16_5_overall1, data_our_ANN_16_5_overall2, data_our_ANN_16_5_cost, data_our_ANN_16_5_pref, data_our_ANN_16_5_brisk, data_our_ANN_16_5_collision, data_our_ANN_16_5_risk, data_our_ANN_16_5_avoid = utils.read_result(path_our_ANN_16_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_16_5_overall1:{}'.format((data_our_ANN_16_5_overall1)))
print('-'*30)
print('data_our_ANN_16_5_overall1:{}'.format(np.mean(data_our_ANN_16_5_overall1)))
print('data_our_ANN_16_5_cost:{}'.format(np.mean(data_our_ANN_16_5_cost)))
print('data_our_ANN_16_5_pref:{}'.format(np.mean(data_our_ANN_16_5_pref)))
print('data_our_ANN_16_5_brisk:{}'.format(np.mean(data_our_ANN_16_5_brisk)))
print('data_our_ANN_16_5_collision:{}'.format(np.mean(data_our_ANN_16_5_collision)))
print('data_our_ANN_16_5_risk:{}'.format(np.mean(data_our_ANN_16_5_risk)))
print('data_our_ANN_16_5_avoid:{}'.format(np.mean(data_our_ANN_16_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_100, 8)
# -----------------------------------------
path_our_SVM_100_8 = utils.root_path() + 'results/8/test_SVM_100_8_nonew/'
data_our_SVM_100_8_overall1, data_our_SVM_100_8_overall2, data_our_SVM_100_8_cost, data_our_SVM_100_8_pref, data_our_SVM_100_8_brisk, data_our_SVM_100_8_collision, data_our_SVM_100_8_risk, data_our_SVM_100_8_avoid = utils.read_result(path_our_SVM_100_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_100_8_overall1:{}'.format((data_our_SVM_100_8_overall1)))
print('-'*30)
print('data_our_SVM_100_8_overall1:{}'.format(np.mean(data_our_SVM_100_8_overall1)))
print('data_our_SVM_100_8_cost:{}'.format(np.mean(data_our_SVM_100_8_cost)))
print('data_our_SVM_100_8_pref:{}'.format(np.mean(data_our_SVM_100_8_pref)))
print('data_our_SVM_100_8_brisk:{}'.format(np.mean(data_our_SVM_100_8_brisk)))
print('data_our_SVM_100_8_collision:{}'.format(np.mean(data_our_SVM_100_8_collision)))
print('data_our_SVM_100_8_risk:{}'.format(np.mean(data_our_SVM_100_8_risk)))
print('data_our_SVM_100_8_avoid:{}'.format(np.mean(data_our_SVM_100_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_100, 5)
# -----------------------------------------
path_our_SVM_100_5 = utils.root_path() + 'results/5/test_SVM_100_5_nonew/'
data_our_SVM_100_5_overall1, data_our_SVM_100_5_overall2, data_our_SVM_100_5_cost, data_our_SVM_100_5_pref, data_our_SVM_100_5_brisk, data_our_SVM_100_5_collision, data_our_SVM_100_5_risk, data_our_SVM_100_5_avoid = utils.read_result(path_our_SVM_100_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_100_5_overall1:{}'.format((data_our_SVM_100_5_overall1)))
print('-'*30)
print('data_our_SVM_100_5_overall1:{}'.format(np.mean(data_our_SVM_100_5_overall1)))
print('data_our_SVM_100_5_cost:{}'.format(np.mean(data_our_SVM_100_5_cost)))
print('data_our_SVM_100_5_pref:{}'.format(np.mean(data_our_SVM_100_5_pref)))
print('data_our_SVM_100_5_brisk:{}'.format(np.mean(data_our_SVM_100_5_brisk)))
print('data_our_SVM_100_5_collision:{}'.format(np.mean(data_our_SVM_100_5_collision)))
print('data_our_SVM_100_5_risk:{}'.format(np.mean(data_our_SVM_100_5_risk)))
print('data_our_SVM_100_5_avoid:{}'.format(np.mean(data_our_SVM_100_5_avoid)))

# -----------------------------------------
# data processing: Our (SVM_83, 8)
# -----------------------------------------
path_our_SVM_83_8 = utils.root_path() + 'results/8/test_SVM_83_8_nonew/'
data_our_SVM_83_8_overall1, data_our_SVM_83_8_overall2, data_our_SVM_83_8_cost, data_our_SVM_83_8_pref, data_our_SVM_83_8_brisk, data_our_SVM_83_8_collision, data_our_SVM_83_8_risk, data_our_SVM_83_8_avoid = utils.read_result(path_our_SVM_83_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_83_8_overall1:{}'.format((data_our_SVM_83_8_overall1)))
print('-'*30)
print('data_our_SVM_83_8_overall1:{}'.format(np.mean(data_our_SVM_83_8_overall1)))
print('data_our_SVM_83_8_cost:{}'.format(np.mean(data_our_SVM_83_8_cost)))
print('data_our_SVM_83_8_pref:{}'.format(np.mean(data_our_SVM_83_8_pref)))
print('data_our_SVM_83_8_brisk:{}'.format(np.mean(data_our_SVM_83_8_brisk)))
print('data_our_SVM_83_8_collision:{}'.format(np.mean(data_our_SVM_83_8_collision)))
print('data_our_SVM_83_8_risk:{}'.format(np.mean(data_our_SVM_83_8_risk)))
print('data_our_SVM_83_8_avoid:{}'.format(np.mean(data_our_SVM_83_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_83, 5)
# -----------------------------------------
path_our_SVM_83_5 = utils.root_path() + 'results/5/test_SVM_83_5_nonew/'
data_our_SVM_83_5_overall1, data_our_SVM_83_5_overall2, data_our_SVM_83_5_cost, data_our_SVM_83_5_pref, data_our_SVM_83_5_brisk, data_our_SVM_83_5_collision, data_our_SVM_83_5_risk, data_our_SVM_83_5_avoid = utils.read_result(path_our_SVM_83_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_83_5_overall1:{}'.format((data_our_SVM_83_5_overall1)))
print('-'*30)
print('data_our_SVM_83_5_overall1:{}'.format(np.mean(data_our_SVM_83_5_overall1)))
print('data_our_SVM_83_5_cost:{}'.format(np.mean(data_our_SVM_83_5_cost)))
print('data_our_SVM_83_5_pref:{}'.format(np.mean(data_our_SVM_83_5_pref)))
print('data_our_SVM_83_5_brisk:{}'.format(np.mean(data_our_SVM_83_5_brisk)))
print('data_our_SVM_83_5_collision:{}'.format(np.mean(data_our_SVM_83_5_collision)))
print('data_our_SVM_83_5_risk:{}'.format(np.mean(data_our_SVM_83_5_risk)))
print('data_our_SVM_83_5_avoid:{}'.format(np.mean(data_our_SVM_83_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_66, 8)
# -----------------------------------------
path_our_SVM_66_8 = utils.root_path() + 'results/8/test_SVM_66_8_nonew/'
data_our_SVM_66_8_overall1, data_our_SVM_66_8_overall2, data_our_SVM_66_8_cost, data_our_SVM_66_8_pref, data_our_SVM_66_8_brisk, data_our_SVM_66_8_collision, data_our_SVM_66_8_risk, data_our_SVM_66_8_avoid = utils.read_result(path_our_SVM_66_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_66_8_overall1:{}'.format((data_our_SVM_66_8_overall1)))
print('-'*30)
print('data_our_SVM_66_8_overall1:{}'.format(np.mean(data_our_SVM_66_8_overall1)))
print('data_our_SVM_66_8_cost:{}'.format(np.mean(data_our_SVM_66_8_cost)))
print('data_our_SVM_66_8_pref:{}'.format(np.mean(data_our_SVM_66_8_pref)))
print('data_our_SVM_66_8_brisk:{}'.format(np.mean(data_our_SVM_66_8_brisk)))
print('data_our_SVM_66_8_collision:{}'.format(np.mean(data_our_SVM_66_8_collision)))
print('data_our_SVM_66_8_risk:{}'.format(np.mean(data_our_SVM_66_8_risk)))
print('data_our_SVM_66_8_avoid:{}'.format(np.mean(data_our_SVM_66_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_66, 5)
# -----------------------------------------
path_our_SVM_66_5 = utils.root_path() + 'results/5/test_SVM_66_5_nonew/'
data_our_SVM_66_5_overall1, data_our_SVM_66_5_overall2, data_our_SVM_66_5_cost, data_our_SVM_66_5_pref, data_our_SVM_66_5_brisk, data_our_SVM_66_5_collision, data_our_SVM_66_5_risk, data_our_SVM_66_5_avoid = utils.read_result(path_our_SVM_66_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_66_5_overall1:{}'.format((data_our_SVM_66_5_overall1)))
print('-'*30)
print('data_our_SVM_66_5_overall1:{}'.format(np.mean(data_our_SVM_66_5_overall1)))
print('data_our_SVM_66_5_cost:{}'.format(np.mean(data_our_SVM_66_5_cost)))
print('data_our_SVM_66_5_pref:{}'.format(np.mean(data_our_SVM_66_5_pref)))
print('data_our_SVM_66_5_brisk:{}'.format(np.mean(data_our_SVM_66_5_brisk)))
print('data_our_SVM_66_5_collision:{}'.format(np.mean(data_our_SVM_66_5_collision)))
print('data_our_SVM_66_5_risk:{}'.format(np.mean(data_our_SVM_66_5_risk)))
print('data_our_SVM_66_5_avoid:{}'.format(np.mean(data_our_SVM_66_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_50, 8)
# -----------------------------------------
path_our_SVM_50_8 = utils.root_path() + 'results/8/test_SVM_50_8_nonew/'
data_our_SVM_50_8_overall1, data_our_SVM_50_8_overall2, data_our_SVM_50_8_cost, data_our_SVM_50_8_pref, data_our_SVM_50_8_brisk, data_our_SVM_50_8_collision, data_our_SVM_50_8_risk, data_our_SVM_50_8_avoid = utils.read_result(path_our_SVM_50_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_50_8_overall1:{}'.format((data_our_SVM_50_8_overall1)))
print('-'*30)
print('data_our_SVM_50_8_overall1:{}'.format(np.mean(data_our_SVM_50_8_overall1)))
print('data_our_SVM_50_8_cost:{}'.format(np.mean(data_our_SVM_50_8_cost)))
print('data_our_SVM_50_8_pref:{}'.format(np.mean(data_our_SVM_50_8_pref)))
print('data_our_SVM_50_8_brisk:{}'.format(np.mean(data_our_SVM_50_8_brisk)))
print('data_our_SVM_50_8_collision:{}'.format(np.mean(data_our_SVM_50_8_collision)))
print('data_our_SVM_50_8_risk:{}'.format(np.mean(data_our_SVM_50_8_risk)))
print('data_our_SVM_50_8_avoid:{}'.format(np.mean(data_our_SVM_50_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_50, 5)
# -----------------------------------------
path_our_SVM_50_5 = utils.root_path() + 'results/5/test_SVM_50_5_nonew/'
data_our_SVM_50_5_overall1, data_our_SVM_50_5_overall2, data_our_SVM_50_5_cost, data_our_SVM_50_5_pref, data_our_SVM_50_5_brisk, data_our_SVM_50_5_collision, data_our_SVM_50_5_risk, data_our_SVM_50_5_avoid = utils.read_result(path_our_SVM_50_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_50_5_overall1:{}'.format((data_our_SVM_50_5_overall1)))
print('-'*30)
print('data_our_SVM_50_5_overall1:{}'.format(np.mean(data_our_SVM_50_5_overall1)))
print('data_our_SVM_50_5_cost:{}'.format(np.mean(data_our_SVM_50_5_cost)))
print('data_our_SVM_50_5_pref:{}'.format(np.mean(data_our_SVM_50_5_pref)))
print('data_our_SVM_50_5_brisk:{}'.format(np.mean(data_our_SVM_50_5_brisk)))
print('data_our_SVM_50_5_collision:{}'.format(np.mean(data_our_SVM_50_5_collision)))
print('data_our_SVM_50_5_risk:{}'.format(np.mean(data_our_SVM_50_5_risk)))
print('data_our_SVM_50_5_avoid:{}'.format(np.mean(data_our_SVM_50_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_33, 8)
# -----------------------------------------
path_our_SVM_33_8 = utils.root_path() + 'results/8/test_SVM_33_8_nonew/'
data_our_SVM_33_8_overall1, data_our_SVM_33_8_overall2, data_our_SVM_33_8_cost, data_our_SVM_33_8_pref, data_our_SVM_33_8_brisk, data_our_SVM_33_8_collision, data_our_SVM_33_8_risk, data_our_SVM_33_8_avoid = utils.read_result(path_our_SVM_33_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_33_8_overall1:{}'.format((data_our_SVM_33_8_overall1)))
print('-'*30)
print('data_our_SVM_33_8_overall1:{}'.format(np.mean(data_our_SVM_33_8_overall1)))
print('data_our_SVM_33_8_cost:{}'.format(np.mean(data_our_SVM_33_8_cost)))
print('data_our_SVM_33_8_pref:{}'.format(np.mean(data_our_SVM_33_8_pref)))
print('data_our_SVM_33_8_brisk:{}'.format(np.mean(data_our_SVM_33_8_brisk)))
print('data_our_SVM_33_8_collision:{}'.format(np.mean(data_our_SVM_33_8_collision)))
print('data_our_SVM_33_8_risk:{}'.format(np.mean(data_our_SVM_33_8_risk)))
print('data_our_SVM_33_8_avoid:{}'.format(np.mean(data_our_SVM_33_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_33, 5)
# -----------------------------------------
path_our_SVM_33_5 = utils.root_path() + 'results/5/test_SVM_33_5_nonew/'
data_our_SVM_33_5_overall1, data_our_SVM_33_5_overall2, data_our_SVM_33_5_cost, data_our_SVM_33_5_pref, data_our_SVM_33_5_brisk, data_our_SVM_33_5_collision, data_our_SVM_33_5_risk, data_our_SVM_33_5_avoid = utils.read_result(path_our_SVM_33_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_33_5_overall1:{}'.format((data_our_SVM_33_5_overall1)))
print('-'*30)
print('data_our_SVM_33_5_overall1:{}'.format(np.mean(data_our_SVM_33_5_overall1)))
print('data_our_SVM_33_5_cost:{}'.format(np.mean(data_our_SVM_33_5_cost)))
print('data_our_SVM_33_5_pref:{}'.format(np.mean(data_our_SVM_33_5_pref)))
print('data_our_SVM_33_5_brisk:{}'.format(np.mean(data_our_SVM_33_5_brisk)))
print('data_our_SVM_33_5_collision:{}'.format(np.mean(data_our_SVM_33_5_collision)))
print('data_our_SVM_33_5_risk:{}'.format(np.mean(data_our_SVM_33_5_risk)))
print('data_our_SVM_33_5_avoid:{}'.format(np.mean(data_our_SVM_33_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_16, 8)
# -----------------------------------------
path_our_SVM_16_8 = utils.root_path() + 'results/8/test_SVM_16_8_nonew/'
data_our_SVM_16_8_overall1, data_our_SVM_16_8_overall2, data_our_SVM_16_8_cost, data_our_SVM_16_8_pref, data_our_SVM_16_8_brisk, data_our_SVM_16_8_collision, data_our_SVM_16_8_risk, data_our_SVM_16_8_avoid = utils.read_result(path_our_SVM_16_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_16_8_overall1:{}'.format((data_our_SVM_16_8_overall1)))
print('-'*30)
print('data_our_SVM_16_8_overall1:{}'.format(np.mean(data_our_SVM_16_8_overall1)))
print('data_our_SVM_16_8_cost:{}'.format(np.mean(data_our_SVM_16_8_cost)))
print('data_our_SVM_16_8_pref:{}'.format(np.mean(data_our_SVM_16_8_pref)))
print('data_our_SVM_16_8_brisk:{}'.format(np.mean(data_our_SVM_16_8_brisk)))
print('data_our_SVM_16_8_collision:{}'.format(np.mean(data_our_SVM_16_8_collision)))
print('data_our_SVM_16_8_risk:{}'.format(np.mean(data_our_SVM_16_8_risk)))
print('data_our_SVM_16_8_avoid:{}'.format(np.mean(data_our_SVM_16_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_16, 5)
# -----------------------------------------
path_our_SVM_16_5 = utils.root_path() + 'results/5/test_SVM_16_5_nonew/'
data_our_SVM_16_5_overall1, data_our_SVM_16_5_overall2, data_our_SVM_16_5_cost, data_our_SVM_16_5_pref, data_our_SVM_16_5_brisk, data_our_SVM_16_5_collision, data_our_SVM_16_5_risk, data_our_SVM_16_5_avoid = utils.read_result(path_our_SVM_16_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_16_5_overall1:{}'.format((data_our_SVM_16_5_overall1)))
print('-'*30)
print('data_our_SVM_16_5_overall1:{}'.format(np.mean(data_our_SVM_16_5_overall1)))
print('data_our_SVM_16_5_cost:{}'.format(np.mean(data_our_SVM_16_5_cost)))
print('data_our_SVM_16_5_pref:{}'.format(np.mean(data_our_SVM_16_5_pref)))
print('data_our_SVM_16_5_brisk:{}'.format(np.mean(data_our_SVM_16_5_brisk)))
print('data_our_SVM_16_5_collision:{}'.format(np.mean(data_our_SVM_16_5_collision)))
print('data_our_SVM_16_5_risk:{}'.format(np.mean(data_our_SVM_16_5_risk)))
print('data_our_SVM_16_5_avoid:{}'.format(np.mean(data_our_SVM_16_5_avoid)))

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X = np.array([1, 2, 3, 4, 5, 6])
labels = ['16.7%', '33.3%', '50%', '66.7%', '83.3%', '100%']

# -----------------------------------------
# Y values (ANN_100, 8)
# -----------------------------------------
Y_our_ANN_100_8 = [np.mean(data_our_ANN_100_8_overall1) + 8.0]
Y_our_ANN_100_8_std = [np.std(data_our_ANN_100_8_overall1) / 2.]
Y_our_ANN_100_5 = [np.mean(data_our_ANN_100_5_overall1)  + 8.0]
Y_our_ANN_100_5_std = [np.std(data_our_ANN_100_5_overall1) / 2.]


# -----------------------------------------
# Y values (ANN_83, 8)
# -----------------------------------------
Y_our_ANN_83_8 = [np.mean(data_our_ANN_83_8_overall1)]
Y_our_ANN_83_8_std = [np.std(data_our_ANN_83_8_overall1) / 2.]
Y_our_ANN_83_5 = [np.mean(data_our_ANN_83_5_overall1)]
Y_our_ANN_83_5_std = [np.std(data_our_ANN_83_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_66, 8)
# -----------------------------------------
Y_our_ANN_66_8 = [np.mean(data_our_ANN_66_8_overall1)]
Y_our_ANN_66_8_std = [np.std(data_our_ANN_66_8_overall1) / 2.]
Y_our_ANN_66_5 = [np.mean(data_our_ANN_66_5_overall1)]
Y_our_ANN_66_5_std = [np.std(data_our_ANN_66_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_50, 8)
# -----------------------------------------
Y_our_ANN_50_8 = [np.mean(data_our_ANN_50_8_overall1)]
Y_our_ANN_50_8_std = [np.std(data_our_ANN_50_8_overall1) / 2.]
Y_our_ANN_50_5 = [np.mean(data_our_ANN_50_5_overall1)]
Y_our_ANN_50_5_std = [np.std(data_our_ANN_50_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_33, 8)
# -----------------------------------------
Y_our_ANN_33_8 = [np.mean(data_our_ANN_33_8_overall1)]
Y_our_ANN_33_8_std = [np.std(data_our_ANN_33_8_overall1) / 2.]
Y_our_ANN_33_5 = [np.mean(data_our_ANN_33_5_overall1)]
Y_our_ANN_33_5_std = [np.std(data_our_ANN_33_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_16, 8)
# -----------------------------------------
Y_our_ANN_16_8 = [np.mean(data_our_ANN_16_8_overall1)]
Y_our_ANN_16_8_std = [np.std(data_our_ANN_16_8_overall1) / 2.]
Y_our_ANN_16_5 = [np.mean(data_our_ANN_16_5_overall1)]
Y_our_ANN_16_5_std = [np.std(data_our_ANN_16_5_overall1) / 2.]


# -----------------------------------------
# Y values (SVM_100, 8)
# -----------------------------------------
Y_our_SVM_100_8 = [np.mean(data_our_SVM_100_8_overall1)]
Y_our_SVM_100_8_std = [np.std(data_our_SVM_100_8_overall1) / 2.]
Y_our_SVM_100_5 = [np.mean(data_our_SVM_100_5_overall1)]
Y_our_SVM_100_5_std = [np.std(data_our_SVM_100_5_overall1) / 2.]


# -----------------------------------------
# Y values (SVM_83, 8)
# -----------------------------------------
Y_our_SVM_83_8 = [np.mean(data_our_SVM_83_8_overall1)]
Y_our_SVM_83_8_std = [np.std(data_our_SVM_83_8_overall1) / 2.]
Y_our_SVM_83_5 = [np.mean(data_our_SVM_83_5_overall1)]
Y_our_SVM_83_5_std = [np.std(data_our_SVM_83_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_66, 8)
# -----------------------------------------
Y_our_SVM_66_8 = [np.mean(data_our_SVM_66_8_overall1)]
Y_our_SVM_66_8_std = [np.std(data_our_SVM_66_8_overall1) / 2.]
Y_our_SVM_66_5 = [np.mean(data_our_SVM_66_5_overall1)]
Y_our_SVM_66_5_std = [np.std(data_our_SVM_66_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_50, 8)
# -----------------------------------------
Y_our_SVM_50_8 = [np.mean(data_our_SVM_50_8_overall1)]
Y_our_SVM_50_8_std = [np.std(data_our_SVM_50_8_overall1) / 2.]
Y_our_SVM_50_5 = [np.mean(data_our_SVM_50_5_overall1)]
Y_our_SVM_50_5_std = [np.std(data_our_SVM_50_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_33, 8)
# -----------------------------------------
Y_our_SVM_33_8 = [np.mean(data_our_SVM_33_8_overall1)]
Y_our_SVM_33_8_std = [np.std(data_our_SVM_33_8_overall1) / 2.]
Y_our_SVM_33_5 = [np.mean(data_our_SVM_33_5_overall1)]
Y_our_SVM_33_5_std = [np.std(data_our_SVM_33_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_16, 8)
# -----------------------------------------
Y_our_SVM_16_8 = [np.mean(data_our_SVM_16_8_overall1)]
Y_our_SVM_16_8_std = [np.std(data_our_SVM_16_8_overall1) / 2.]
Y_our_SVM_16_5 = [np.mean(data_our_SVM_16_5_overall1)]
Y_our_SVM_16_5_std = [np.std(data_our_SVM_16_5_overall1) / 2.]


Y_ANN_8 = [Y_our_ANN_16_8, Y_our_ANN_33_8, Y_our_ANN_50_8, Y_our_ANN_66_8, Y_our_ANN_83_8, Y_our_ANN_100_8]
Y_ANN_8_std = [Y_our_ANN_16_8_std, Y_our_ANN_33_8_std, Y_our_ANN_50_8_std, Y_our_ANN_66_8_std, Y_our_ANN_83_8_std, Y_our_ANN_100_8_std]

Y_ANN_5 = [Y_our_ANN_16_5, Y_our_ANN_33_5, Y_our_ANN_50_5, Y_our_ANN_66_5, Y_our_ANN_83_5, Y_our_ANN_100_5]
Y_ANN_5_std = [Y_our_ANN_16_5_std, Y_our_ANN_33_5_std, Y_our_ANN_50_5_std, Y_our_ANN_66_5_std, Y_our_ANN_83_5_std, Y_our_ANN_100_5_std]

Y_SVM_8 = [Y_our_SVM_16_8, Y_our_SVM_33_8, Y_our_SVM_50_8, Y_our_SVM_66_8, Y_our_SVM_83_8, Y_our_SVM_100_8]
Y_SVM_8_std = [Y_our_SVM_16_8_std, Y_our_SVM_33_8_std, Y_our_SVM_50_8_std, Y_our_SVM_66_8_std, Y_our_SVM_83_8_std, Y_our_SVM_100_8_std]

Y_SVM_5 = [Y_our_SVM_16_5, Y_our_SVM_33_5, Y_our_SVM_50_5, Y_our_SVM_66_5, Y_our_SVM_83_5, Y_our_SVM_100_5]
Y_SVM_5_std = [Y_our_SVM_16_5_std, Y_our_SVM_33_5_std, Y_our_SVM_50_5_std, Y_our_SVM_66_5_std, Y_our_SVM_83_5_std, Y_our_SVM_100_5_std]

# -----------------------------------------
# start plotting
# -----------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(params['figure_width'], params['figure_height']))

ax[0][0].plot(X, Y_ANN_8, marker=params['marker'][0], color=params['color'][0], label='GLAD-Net', linewidth=params['line_width'], linestyle=params['line_style'][0], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_ANN_8 - Y_ANN_8_std, Y_ANN_8 + Y_ANN_8_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[1][0].plot(X, Y_ANN_5, marker=params['marker'][1], color=params['color'][0], label='GLAD-Net', linewidth=params['line_width'], linestyle=params['line_style'][0], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_ANN_5 - Y_ANN_5_std, Y_ANN_5 + Y_ANN_5_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[0][1].plot(X, Y_SVM_8, marker=params['marker'][2], color=params['color'][0], label='GLAD-SVM', linewidth=params['line_width'], linestyle=params['line_style'][1], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_SVM_8 - Y_SVM_8_std, Y_SVM_8 + Y_SVM_8_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[1][1].plot(X, Y_SVM_5, marker=params['marker'][3], color=params['color'][0], label='GLAD-SVM', linewidth=params['line_width'], linestyle=params['line_style'][1], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_SVM_8 - Y_SVM_8_std, Y_SVM_8 + Y_SVM_8_std, alpha=params['fill_transparent'], color=params['color'][0])

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------
# data processing: Our (ANN_100, 8)
# -----------------------------------------
path_our_ANN_100_8 = utils.root_path() + 'results/8/test_ANN_100_8_new/'
data_our_ANN_100_8_overall1, data_our_ANN_100_8_overall2, data_our_ANN_100_8_cost, data_our_ANN_100_8_pref, data_our_ANN_100_8_brisk, data_our_ANN_100_8_collision, data_our_ANN_100_8_risk, data_our_ANN_100_8_avoid = utils.read_result(path_our_ANN_100_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])

if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_100_8_overall1:{}'.format((data_our_ANN_100_8_overall1)))
print('-'*30)
print('data_our_ANN_100_8_overall1:{}'.format(np.mean(data_our_ANN_100_8_overall1)))
print('data_our_ANN_100_8_cost:{}'.format(np.mean(data_our_ANN_100_8_cost)))
print('data_our_ANN_100_8_pref:{}'.format(np.mean(data_our_ANN_100_8_pref)))
print('data_our_ANN_100_8_brisk:{}'.format(np.mean(data_our_ANN_100_8_brisk)))
print('data_our_ANN_100_8_collision:{}'.format(np.mean(data_our_ANN_100_8_collision)))
print('data_our_ANN_100_8_risk:{}'.format(np.mean(data_our_ANN_100_8_risk)))
print('data_our_ANN_100_8_avoid:{}'.format(np.mean(data_our_ANN_100_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_100, 5)
# -----------------------------------------
path_our_ANN_100_5 = utils.root_path() + 'results/5/test_ANN_100_5_new/'
data_our_ANN_100_5_overall1, data_our_ANN_100_5_overall2, data_our_ANN_100_5_cost, data_our_ANN_100_5_pref, data_our_ANN_100_5_brisk, data_our_ANN_100_5_collision, data_our_ANN_100_5_risk, data_our_ANN_100_5_avoid = utils.read_result(path_our_ANN_100_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])

if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_100_5_overall1:{}'.format((data_our_ANN_100_5_overall1)))
print('-'*30)
print('data_our_ANN_100_5_overall1:{}'.format(np.mean(data_our_ANN_100_5_overall1)))
print('data_our_ANN_100_5_cost:{}'.format(np.mean(data_our_ANN_100_5_cost)))
print('data_our_ANN_100_5_pref:{}'.format(np.mean(data_our_ANN_100_5_pref)))
print('data_our_ANN_100_5_brisk:{}'.format(np.mean(data_our_ANN_100_5_brisk)))
print('data_our_ANN_100_5_collision:{}'.format(np.mean(data_our_ANN_100_5_collision)))
print('data_our_ANN_100_5_risk:{}'.format(np.mean(data_our_ANN_100_5_risk)))
print('data_our_ANN_100_5_avoid:{}'.format(np.mean(data_our_ANN_100_5_avoid)))

# -----------------------------------------
# data processing: Our (ANN_83, 8)
# -----------------------------------------
path_our_ANN_83_8 = utils.root_path() + 'results/8/test_ANN_83_8_new/'
data_our_ANN_83_8_overall1, data_our_ANN_83_8_overall2, data_our_ANN_83_8_cost, data_our_ANN_83_8_pref, data_our_ANN_83_8_brisk, data_our_ANN_83_8_collision, data_our_ANN_83_8_risk, data_our_ANN_83_8_avoid = utils.read_result(path_our_ANN_83_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_83_8_overall1:{}'.format((data_our_ANN_83_8_overall1)))
print('-'*30)
print('data_our_ANN_83_8_overall1:{}'.format(np.mean(data_our_ANN_83_8_overall1)))
print('data_our_ANN_83_8_cost:{}'.format(np.mean(data_our_ANN_83_8_cost)))
print('data_our_ANN_83_8_pref:{}'.format(np.mean(data_our_ANN_83_8_pref)))
print('data_our_ANN_83_8_brisk:{}'.format(np.mean(data_our_ANN_83_8_brisk)))
print('data_our_ANN_83_8_collision:{}'.format(np.mean(data_our_ANN_83_8_collision)))
print('data_our_ANN_83_8_risk:{}'.format(np.mean(data_our_ANN_83_8_risk)))
print('data_our_ANN_83_8_avoid:{}'.format(np.mean(data_our_ANN_83_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_83, 5)
# -----------------------------------------
path_our_ANN_83_5 = utils.root_path() + 'results/5/test_ANN_83_5_new/'
data_our_ANN_83_5_overall1, data_our_ANN_83_5_overall2, data_our_ANN_83_5_cost, data_our_ANN_83_5_pref, data_our_ANN_83_5_brisk, data_our_ANN_83_5_collision, data_our_ANN_83_5_risk, data_our_ANN_83_5_avoid = utils.read_result(path_our_ANN_83_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_83_5_overall1:{}'.format((data_our_ANN_83_5_overall1)))
print('-'*30)
print('data_our_ANN_83_5_overall1:{}'.format(np.mean(data_our_ANN_83_5_overall1)))
print('data_our_ANN_83_5_cost:{}'.format(np.mean(data_our_ANN_83_5_cost)))
print('data_our_ANN_83_5_pref:{}'.format(np.mean(data_our_ANN_83_5_pref)))
print('data_our_ANN_83_5_brisk:{}'.format(np.mean(data_our_ANN_83_5_brisk)))
print('data_our_ANN_83_5_collision:{}'.format(np.mean(data_our_ANN_83_5_collision)))
print('data_our_ANN_83_5_risk:{}'.format(np.mean(data_our_ANN_83_5_risk)))
print('data_our_ANN_83_5_avoid:{}'.format(np.mean(data_our_ANN_83_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_66, 8)
# -----------------------------------------
path_our_ANN_66_8 = utils.root_path() + 'results/8/test_ANN_66_8_new/'
data_our_ANN_66_8_overall1, data_our_ANN_66_8_overall2, data_our_ANN_66_8_cost, data_our_ANN_66_8_pref, data_our_ANN_66_8_brisk, data_our_ANN_66_8_collision, data_our_ANN_66_8_risk, data_our_ANN_66_8_avoid = utils.read_result(path_our_ANN_66_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_66_8_overall1:{}'.format((data_our_ANN_66_8_overall1)))
print('-'*30)
print('data_our_ANN_66_8_overall1:{}'.format(np.mean(data_our_ANN_66_8_overall1)))
print('data_our_ANN_66_8_cost:{}'.format(np.mean(data_our_ANN_66_8_cost)))
print('data_our_ANN_66_8_pref:{}'.format(np.mean(data_our_ANN_66_8_pref)))
print('data_our_ANN_66_8_brisk:{}'.format(np.mean(data_our_ANN_66_8_brisk)))
print('data_our_ANN_66_8_collision:{}'.format(np.mean(data_our_ANN_66_8_collision)))
print('data_our_ANN_66_8_risk:{}'.format(np.mean(data_our_ANN_66_8_risk)))
print('data_our_ANN_66_8_avoid:{}'.format(np.mean(data_our_ANN_66_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_66, 5)
# -----------------------------------------
path_our_ANN_66_5 = utils.root_path() + 'results/5/test_ANN_66_5_new/'
data_our_ANN_66_5_overall1, data_our_ANN_66_5_overall2, data_our_ANN_66_5_cost, data_our_ANN_66_5_pref, data_our_ANN_66_5_brisk, data_our_ANN_66_5_collision, data_our_ANN_66_5_risk, data_our_ANN_66_5_avoid = utils.read_result(path_our_ANN_66_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_66_5_overall1:{}'.format((data_our_ANN_66_5_overall1)))
print('-'*30)
print('data_our_ANN_66_5_overall1:{}'.format(np.mean(data_our_ANN_66_5_overall1)))
print('data_our_ANN_66_5_cost:{}'.format(np.mean(data_our_ANN_66_5_cost)))
print('data_our_ANN_66_5_pref:{}'.format(np.mean(data_our_ANN_66_5_pref)))
print('data_our_ANN_66_5_brisk:{}'.format(np.mean(data_our_ANN_66_5_brisk)))
print('data_our_ANN_66_5_collision:{}'.format(np.mean(data_our_ANN_66_5_collision)))
print('data_our_ANN_66_5_risk:{}'.format(np.mean(data_our_ANN_66_5_risk)))
print('data_our_ANN_66_5_avoid:{}'.format(np.mean(data_our_ANN_66_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_50, 8)
# -----------------------------------------
path_our_ANN_50_8 = utils.root_path() + 'results/8/test_ANN_50_8_new/'
data_our_ANN_50_8_overall1, data_our_ANN_50_8_overall2, data_our_ANN_50_8_cost, data_our_ANN_50_8_pref, data_our_ANN_50_8_brisk, data_our_ANN_50_8_collision, data_our_ANN_50_8_risk, data_our_ANN_50_8_avoid = utils.read_result(path_our_ANN_50_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_50_8_overall1:{}'.format((data_our_ANN_50_8_overall1)))
print('-'*30)
print('data_our_ANN_50_8_overall1:{}'.format(np.mean(data_our_ANN_50_8_overall1)))
print('data_our_ANN_50_8_cost:{}'.format(np.mean(data_our_ANN_50_8_cost)))
print('data_our_ANN_50_8_pref:{}'.format(np.mean(data_our_ANN_50_8_pref)))
print('data_our_ANN_50_8_brisk:{}'.format(np.mean(data_our_ANN_50_8_brisk)))
print('data_our_ANN_50_8_collision:{}'.format(np.mean(data_our_ANN_50_8_collision)))
print('data_our_ANN_50_8_risk:{}'.format(np.mean(data_our_ANN_50_8_risk)))
print('data_our_ANN_50_8_avoid:{}'.format(np.mean(data_our_ANN_50_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_50, 5)
# -----------------------------------------
path_our_ANN_50_5 = utils.root_path() + 'results/5/test_ANN_50_5_new/'
data_our_ANN_50_5_overall1, data_our_ANN_50_5_overall2, data_our_ANN_50_5_cost, data_our_ANN_50_5_pref, data_our_ANN_50_5_brisk, data_our_ANN_50_5_collision, data_our_ANN_50_5_risk, data_our_ANN_50_5_avoid = utils.read_result(path_our_ANN_50_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_50_5_overall1:{}'.format((data_our_ANN_50_5_overall1)))
print('-'*30)
print('data_our_ANN_50_5_overall1:{}'.format(np.mean(data_our_ANN_50_5_overall1)))
print('data_our_ANN_50_5_cost:{}'.format(np.mean(data_our_ANN_50_5_cost)))
print('data_our_ANN_50_5_pref:{}'.format(np.mean(data_our_ANN_50_5_pref)))
print('data_our_ANN_50_5_brisk:{}'.format(np.mean(data_our_ANN_50_5_brisk)))
print('data_our_ANN_50_5_collision:{}'.format(np.mean(data_our_ANN_50_5_collision)))
print('data_our_ANN_50_5_risk:{}'.format(np.mean(data_our_ANN_50_5_risk)))
print('data_our_ANN_50_5_avoid:{}'.format(np.mean(data_our_ANN_50_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_33, 8)
# -----------------------------------------
path_our_ANN_33_8 = utils.root_path() + 'results/8/test_ANN_33_8_new/'
data_our_ANN_33_8_overall1, data_our_ANN_33_8_overall2, data_our_ANN_33_8_cost, data_our_ANN_33_8_pref, data_our_ANN_33_8_brisk, data_our_ANN_33_8_collision, data_our_ANN_33_8_risk, data_our_ANN_33_8_avoid = utils.read_result(path_our_ANN_33_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_33_8_overall1:{}'.format((data_our_ANN_33_8_overall1)))
print('-'*30)
print('data_our_ANN_33_8_overall1:{}'.format(np.mean(data_our_ANN_33_8_overall1)))
print('data_our_ANN_33_8_cost:{}'.format(np.mean(data_our_ANN_33_8_cost)))
print('data_our_ANN_33_8_pref:{}'.format(np.mean(data_our_ANN_33_8_pref)))
print('data_our_ANN_33_8_brisk:{}'.format(np.mean(data_our_ANN_33_8_brisk)))
print('data_our_ANN_33_8_collision:{}'.format(np.mean(data_our_ANN_33_8_collision)))
print('data_our_ANN_33_8_risk:{}'.format(np.mean(data_our_ANN_33_8_risk)))
print('data_our_ANN_33_8_avoid:{}'.format(np.mean(data_our_ANN_33_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_33, 5)
# -----------------------------------------
path_our_ANN_33_5 = utils.root_path() + 'results/5/test_ANN_33_5_new/'
data_our_ANN_33_5_overall1, data_our_ANN_33_5_overall2, data_our_ANN_33_5_cost, data_our_ANN_33_5_pref, data_our_ANN_33_5_brisk, data_our_ANN_33_5_collision, data_our_ANN_33_5_risk, data_our_ANN_33_5_avoid = utils.read_result(path_our_ANN_33_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_33_5_overall1:{}'.format((data_our_ANN_33_5_overall1)))
print('-'*30)
print('data_our_ANN_33_5_overall1:{}'.format(np.mean(data_our_ANN_33_5_overall1)))
print('data_our_ANN_33_5_cost:{}'.format(np.mean(data_our_ANN_33_5_cost)))
print('data_our_ANN_33_5_pref:{}'.format(np.mean(data_our_ANN_33_5_pref)))
print('data_our_ANN_33_5_brisk:{}'.format(np.mean(data_our_ANN_33_5_brisk)))
print('data_our_ANN_33_5_collision:{}'.format(np.mean(data_our_ANN_33_5_collision)))
print('data_our_ANN_33_5_risk:{}'.format(np.mean(data_our_ANN_33_5_risk)))
print('data_our_ANN_33_5_avoid:{}'.format(np.mean(data_our_ANN_33_5_avoid)))


# -----------------------------------------
# data processing: Our (ANN_16, 8)
# -----------------------------------------
path_our_ANN_16_8 = utils.root_path() + 'results/8/test_ANN_16_8_new/'
data_our_ANN_16_8_overall1, data_our_ANN_16_8_overall2, data_our_ANN_16_8_cost, data_our_ANN_16_8_pref, data_our_ANN_16_8_brisk, data_our_ANN_16_8_collision, data_our_ANN_16_8_risk, data_our_ANN_16_8_avoid = utils.read_result(path_our_ANN_16_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_16_8_overall1:{}'.format((data_our_ANN_16_8_overall1)))
print('-'*30)
print('data_our_ANN_16_8_overall1:{}'.format(np.mean(data_our_ANN_16_8_overall1)))
print('data_our_ANN_16_8_cost:{}'.format(np.mean(data_our_ANN_16_8_cost)))
print('data_our_ANN_16_8_pref:{}'.format(np.mean(data_our_ANN_16_8_pref)))
print('data_our_ANN_16_8_brisk:{}'.format(np.mean(data_our_ANN_16_8_brisk)))
print('data_our_ANN_16_8_collision:{}'.format(np.mean(data_our_ANN_16_8_collision)))
print('data_our_ANN_16_8_risk:{}'.format(np.mean(data_our_ANN_16_8_risk)))
print('data_our_ANN_16_8_avoid:{}'.format(np.mean(data_our_ANN_16_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN_16, 5)
# -----------------------------------------
path_our_ANN_16_5 = utils.root_path() + 'results/5/test_ANN_16_5_new/'
data_our_ANN_16_5_overall1, data_our_ANN_16_5_overall2, data_our_ANN_16_5_cost, data_our_ANN_16_5_pref, data_our_ANN_16_5_brisk, data_our_ANN_16_5_collision, data_our_ANN_16_5_risk, data_our_ANN_16_5_avoid = utils.read_result(path_our_ANN_16_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_16_5_overall1:{}'.format((data_our_ANN_16_5_overall1)))
print('-'*30)
print('data_our_ANN_16_5_overall1:{}'.format(np.mean(data_our_ANN_16_5_overall1)))
print('data_our_ANN_16_5_cost:{}'.format(np.mean(data_our_ANN_16_5_cost)))
print('data_our_ANN_16_5_pref:{}'.format(np.mean(data_our_ANN_16_5_pref)))
print('data_our_ANN_16_5_brisk:{}'.format(np.mean(data_our_ANN_16_5_brisk)))
print('data_our_ANN_16_5_collision:{}'.format(np.mean(data_our_ANN_16_5_collision)))
print('data_our_ANN_16_5_risk:{}'.format(np.mean(data_our_ANN_16_5_risk)))
print('data_our_ANN_16_5_avoid:{}'.format(np.mean(data_our_ANN_16_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_100, 8)
# -----------------------------------------
path_our_SVM_100_8 = utils.root_path() + 'results/8/test_SVM_100_8_new/'
data_our_SVM_100_8_overall1, data_our_SVM_100_8_overall2, data_our_SVM_100_8_cost, data_our_SVM_100_8_pref, data_our_SVM_100_8_brisk, data_our_SVM_100_8_collision, data_our_SVM_100_8_risk, data_our_SVM_100_8_avoid = utils.read_result(path_our_SVM_100_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_100_8_overall1:{}'.format((data_our_SVM_100_8_overall1)))
print('-'*30)
print('data_our_SVM_100_8_overall1:{}'.format(np.mean(data_our_SVM_100_8_overall1)))
print('data_our_SVM_100_8_cost:{}'.format(np.mean(data_our_SVM_100_8_cost)))
print('data_our_SVM_100_8_pref:{}'.format(np.mean(data_our_SVM_100_8_pref)))
print('data_our_SVM_100_8_brisk:{}'.format(np.mean(data_our_SVM_100_8_brisk)))
print('data_our_SVM_100_8_collision:{}'.format(np.mean(data_our_SVM_100_8_collision)))
print('data_our_SVM_100_8_risk:{}'.format(np.mean(data_our_SVM_100_8_risk)))
print('data_our_SVM_100_8_avoid:{}'.format(np.mean(data_our_SVM_100_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_100, 5)
# -----------------------------------------
path_our_SVM_100_5 = utils.root_path() + 'results/5/test_SVM_100_5_new/'
data_our_SVM_100_5_overall1, data_our_SVM_100_5_overall2, data_our_SVM_100_5_cost, data_our_SVM_100_5_pref, data_our_SVM_100_5_brisk, data_our_SVM_100_5_collision, data_our_SVM_100_5_risk, data_our_SVM_100_5_avoid = utils.read_result(path_our_SVM_100_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_100_5_overall1:{}'.format((data_our_SVM_100_5_overall1)))
print('-'*30)
print('data_our_SVM_100_5_overall1:{}'.format(np.mean(data_our_SVM_100_5_overall1)))
print('data_our_SVM_100_5_cost:{}'.format(np.mean(data_our_SVM_100_5_cost)))
print('data_our_SVM_100_5_pref:{}'.format(np.mean(data_our_SVM_100_5_pref)))
print('data_our_SVM_100_5_brisk:{}'.format(np.mean(data_our_SVM_100_5_brisk)))
print('data_our_SVM_100_5_collision:{}'.format(np.mean(data_our_SVM_100_5_collision)))
print('data_our_SVM_100_5_risk:{}'.format(np.mean(data_our_SVM_100_5_risk)))
print('data_our_SVM_100_5_avoid:{}'.format(np.mean(data_our_SVM_100_5_avoid)))

# -----------------------------------------
# data processing: Our (SVM_83, 8)
# -----------------------------------------
path_our_SVM_83_8 = utils.root_path() + 'results/8/test_SVM_83_8_new/'
data_our_SVM_83_8_overall1, data_our_SVM_83_8_overall2, data_our_SVM_83_8_cost, data_our_SVM_83_8_pref, data_our_SVM_83_8_brisk, data_our_SVM_83_8_collision, data_our_SVM_83_8_risk, data_our_SVM_83_8_avoid = utils.read_result(path_our_SVM_83_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_83_8_overall1:{}'.format((data_our_SVM_83_8_overall1)))
print('-'*30)
print('data_our_SVM_83_8_overall1:{}'.format(np.mean(data_our_SVM_83_8_overall1)))
print('data_our_SVM_83_8_cost:{}'.format(np.mean(data_our_SVM_83_8_cost)))
print('data_our_SVM_83_8_pref:{}'.format(np.mean(data_our_SVM_83_8_pref)))
print('data_our_SVM_83_8_brisk:{}'.format(np.mean(data_our_SVM_83_8_brisk)))
print('data_our_SVM_83_8_collision:{}'.format(np.mean(data_our_SVM_83_8_collision)))
print('data_our_SVM_83_8_risk:{}'.format(np.mean(data_our_SVM_83_8_risk)))
print('data_our_SVM_83_8_avoid:{}'.format(np.mean(data_our_SVM_83_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_83, 5)
# -----------------------------------------
path_our_SVM_83_5 = utils.root_path() + 'results/5/test_SVM_83_5_new/'
data_our_SVM_83_5_overall1, data_our_SVM_83_5_overall2, data_our_SVM_83_5_cost, data_our_SVM_83_5_pref, data_our_SVM_83_5_brisk, data_our_SVM_83_5_collision, data_our_SVM_83_5_risk, data_our_SVM_83_5_avoid = utils.read_result(path_our_SVM_83_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_83_5_overall1:{}'.format((data_our_SVM_83_5_overall1)))
print('-'*30)
print('data_our_SVM_83_5_overall1:{}'.format(np.mean(data_our_SVM_83_5_overall1)))
print('data_our_SVM_83_5_cost:{}'.format(np.mean(data_our_SVM_83_5_cost)))
print('data_our_SVM_83_5_pref:{}'.format(np.mean(data_our_SVM_83_5_pref)))
print('data_our_SVM_83_5_brisk:{}'.format(np.mean(data_our_SVM_83_5_brisk)))
print('data_our_SVM_83_5_collision:{}'.format(np.mean(data_our_SVM_83_5_collision)))
print('data_our_SVM_83_5_risk:{}'.format(np.mean(data_our_SVM_83_5_risk)))
print('data_our_SVM_83_5_avoid:{}'.format(np.mean(data_our_SVM_83_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_66, 8)
# -----------------------------------------
path_our_SVM_66_8 = utils.root_path() + 'results/8/test_SVM_66_8_new/'
data_our_SVM_66_8_overall1, data_our_SVM_66_8_overall2, data_our_SVM_66_8_cost, data_our_SVM_66_8_pref, data_our_SVM_66_8_brisk, data_our_SVM_66_8_collision, data_our_SVM_66_8_risk, data_our_SVM_66_8_avoid = utils.read_result(path_our_SVM_66_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_66_8_overall1:{}'.format((data_our_SVM_66_8_overall1)))
print('-'*30)
print('data_our_SVM_66_8_overall1:{}'.format(np.mean(data_our_SVM_66_8_overall1)))
print('data_our_SVM_66_8_cost:{}'.format(np.mean(data_our_SVM_66_8_cost)))
print('data_our_SVM_66_8_pref:{}'.format(np.mean(data_our_SVM_66_8_pref)))
print('data_our_SVM_66_8_brisk:{}'.format(np.mean(data_our_SVM_66_8_brisk)))
print('data_our_SVM_66_8_collision:{}'.format(np.mean(data_our_SVM_66_8_collision)))
print('data_our_SVM_66_8_risk:{}'.format(np.mean(data_our_SVM_66_8_risk)))
print('data_our_SVM_66_8_avoid:{}'.format(np.mean(data_our_SVM_66_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_66, 5)
# -----------------------------------------
path_our_SVM_66_5 = utils.root_path() + 'results/5/test_SVM_66_5_new/'
data_our_SVM_66_5_overall1, data_our_SVM_66_5_overall2, data_our_SVM_66_5_cost, data_our_SVM_66_5_pref, data_our_SVM_66_5_brisk, data_our_SVM_66_5_collision, data_our_SVM_66_5_risk, data_our_SVM_66_5_avoid = utils.read_result(path_our_SVM_66_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_66_5_overall1:{}'.format((data_our_SVM_66_5_overall1)))
print('-'*30)
print('data_our_SVM_66_5_overall1:{}'.format(np.mean(data_our_SVM_66_5_overall1)))
print('data_our_SVM_66_5_cost:{}'.format(np.mean(data_our_SVM_66_5_cost)))
print('data_our_SVM_66_5_pref:{}'.format(np.mean(data_our_SVM_66_5_pref)))
print('data_our_SVM_66_5_brisk:{}'.format(np.mean(data_our_SVM_66_5_brisk)))
print('data_our_SVM_66_5_collision:{}'.format(np.mean(data_our_SVM_66_5_collision)))
print('data_our_SVM_66_5_risk:{}'.format(np.mean(data_our_SVM_66_5_risk)))
print('data_our_SVM_66_5_avoid:{}'.format(np.mean(data_our_SVM_66_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_50, 8)
# -----------------------------------------
path_our_SVM_50_8 = utils.root_path() + 'results/8/test_SVM_50_8_new/'
data_our_SVM_50_8_overall1, data_our_SVM_50_8_overall2, data_our_SVM_50_8_cost, data_our_SVM_50_8_pref, data_our_SVM_50_8_brisk, data_our_SVM_50_8_collision, data_our_SVM_50_8_risk, data_our_SVM_50_8_avoid = utils.read_result(path_our_SVM_50_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_50_8_overall1:{}'.format((data_our_SVM_50_8_overall1)))
print('-'*30)
print('data_our_SVM_50_8_overall1:{}'.format(np.mean(data_our_SVM_50_8_overall1)))
print('data_our_SVM_50_8_cost:{}'.format(np.mean(data_our_SVM_50_8_cost)))
print('data_our_SVM_50_8_pref:{}'.format(np.mean(data_our_SVM_50_8_pref)))
print('data_our_SVM_50_8_brisk:{}'.format(np.mean(data_our_SVM_50_8_brisk)))
print('data_our_SVM_50_8_collision:{}'.format(np.mean(data_our_SVM_50_8_collision)))
print('data_our_SVM_50_8_risk:{}'.format(np.mean(data_our_SVM_50_8_risk)))
print('data_our_SVM_50_8_avoid:{}'.format(np.mean(data_our_SVM_50_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_50, 5)
# -----------------------------------------
path_our_SVM_50_5 = utils.root_path() + 'results/5/test_SVM_50_5_new/'
data_our_SVM_50_5_overall1, data_our_SVM_50_5_overall2, data_our_SVM_50_5_cost, data_our_SVM_50_5_pref, data_our_SVM_50_5_brisk, data_our_SVM_50_5_collision, data_our_SVM_50_5_risk, data_our_SVM_50_5_avoid = utils.read_result(path_our_SVM_50_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_50_5_overall1:{}'.format((data_our_SVM_50_5_overall1)))
print('-'*30)
print('data_our_SVM_50_5_overall1:{}'.format(np.mean(data_our_SVM_50_5_overall1)))
print('data_our_SVM_50_5_cost:{}'.format(np.mean(data_our_SVM_50_5_cost)))
print('data_our_SVM_50_5_pref:{}'.format(np.mean(data_our_SVM_50_5_pref)))
print('data_our_SVM_50_5_brisk:{}'.format(np.mean(data_our_SVM_50_5_brisk)))
print('data_our_SVM_50_5_collision:{}'.format(np.mean(data_our_SVM_50_5_collision)))
print('data_our_SVM_50_5_risk:{}'.format(np.mean(data_our_SVM_50_5_risk)))
print('data_our_SVM_50_5_avoid:{}'.format(np.mean(data_our_SVM_50_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_33, 8)
# -----------------------------------------
path_our_SVM_33_8 = utils.root_path() + 'results/8/test_SVM_33_8_new/'
data_our_SVM_33_8_overall1, data_our_SVM_33_8_overall2, data_our_SVM_33_8_cost, data_our_SVM_33_8_pref, data_our_SVM_33_8_brisk, data_our_SVM_33_8_collision, data_our_SVM_33_8_risk, data_our_SVM_33_8_avoid = utils.read_result(path_our_SVM_33_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_33_8_overall1:{}'.format((data_our_SVM_33_8_overall1)))
print('-'*30)
print('data_our_SVM_33_8_overall1:{}'.format(np.mean(data_our_SVM_33_8_overall1)))
print('data_our_SVM_33_8_cost:{}'.format(np.mean(data_our_SVM_33_8_cost)))
print('data_our_SVM_33_8_pref:{}'.format(np.mean(data_our_SVM_33_8_pref)))
print('data_our_SVM_33_8_brisk:{}'.format(np.mean(data_our_SVM_33_8_brisk)))
print('data_our_SVM_33_8_collision:{}'.format(np.mean(data_our_SVM_33_8_collision)))
print('data_our_SVM_33_8_risk:{}'.format(np.mean(data_our_SVM_33_8_risk)))
print('data_our_SVM_33_8_avoid:{}'.format(np.mean(data_our_SVM_33_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_33, 5)
# -----------------------------------------
path_our_SVM_33_5 = utils.root_path() + 'results/5/test_SVM_33_5_new/'
data_our_SVM_33_5_overall1, data_our_SVM_33_5_overall2, data_our_SVM_33_5_cost, data_our_SVM_33_5_pref, data_our_SVM_33_5_brisk, data_our_SVM_33_5_collision, data_our_SVM_33_5_risk, data_our_SVM_33_5_avoid = utils.read_result(path_our_SVM_33_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_33_5_overall1:{}'.format((data_our_SVM_33_5_overall1)))
print('-'*30)
print('data_our_SVM_33_5_overall1:{}'.format(np.mean(data_our_SVM_33_5_overall1)))
print('data_our_SVM_33_5_cost:{}'.format(np.mean(data_our_SVM_33_5_cost)))
print('data_our_SVM_33_5_pref:{}'.format(np.mean(data_our_SVM_33_5_pref)))
print('data_our_SVM_33_5_brisk:{}'.format(np.mean(data_our_SVM_33_5_brisk)))
print('data_our_SVM_33_5_collision:{}'.format(np.mean(data_our_SVM_33_5_collision)))
print('data_our_SVM_33_5_risk:{}'.format(np.mean(data_our_SVM_33_5_risk)))
print('data_our_SVM_33_5_avoid:{}'.format(np.mean(data_our_SVM_33_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM_16, 8)
# -----------------------------------------
path_our_SVM_16_8 = utils.root_path() + 'results/8/test_SVM_16_8_new/'
data_our_SVM_16_8_overall1, data_our_SVM_16_8_overall2, data_our_SVM_16_8_cost, data_our_SVM_16_8_pref, data_our_SVM_16_8_brisk, data_our_SVM_16_8_collision, data_our_SVM_16_8_risk, data_our_SVM_16_8_avoid = utils.read_result(path_our_SVM_16_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_16_8_overall1:{}'.format((data_our_SVM_16_8_overall1)))
print('-'*30)
print('data_our_SVM_16_8_overall1:{}'.format(np.mean(data_our_SVM_16_8_overall1)))
print('data_our_SVM_16_8_cost:{}'.format(np.mean(data_our_SVM_16_8_cost)))
print('data_our_SVM_16_8_pref:{}'.format(np.mean(data_our_SVM_16_8_pref)))
print('data_our_SVM_16_8_brisk:{}'.format(np.mean(data_our_SVM_16_8_brisk)))
print('data_our_SVM_16_8_collision:{}'.format(np.mean(data_our_SVM_16_8_collision)))
print('data_our_SVM_16_8_risk:{}'.format(np.mean(data_our_SVM_16_8_risk)))
print('data_our_SVM_16_8_avoid:{}'.format(np.mean(data_our_SVM_16_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM_16, 5)
# -----------------------------------------
path_our_SVM_16_5 = utils.root_path() + 'results/5/test_SVM_16_5_new/'
data_our_SVM_16_5_overall1, data_our_SVM_16_5_overall2, data_our_SVM_16_5_cost, data_our_SVM_16_5_pref, data_our_SVM_16_5_brisk, data_our_SVM_16_5_collision, data_our_SVM_16_5_risk, data_our_SVM_16_5_avoid = utils.read_result(path_our_SVM_16_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_16_5_overall1:{}'.format((data_our_SVM_16_5_overall1)))
print('-'*30)
print('data_our_SVM_16_5_overall1:{}'.format(np.mean(data_our_SVM_16_5_overall1)))
print('data_our_SVM_16_5_cost:{}'.format(np.mean(data_our_SVM_16_5_cost)))
print('data_our_SVM_16_5_pref:{}'.format(np.mean(data_our_SVM_16_5_pref)))
print('data_our_SVM_16_5_brisk:{}'.format(np.mean(data_our_SVM_16_5_brisk)))
print('data_our_SVM_16_5_collision:{}'.format(np.mean(data_our_SVM_16_5_collision)))
print('data_our_SVM_16_5_risk:{}'.format(np.mean(data_our_SVM_16_5_risk)))
print('data_our_SVM_16_5_avoid:{}'.format(np.mean(data_our_SVM_16_5_avoid)))

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X = np.array([1, 2, 3, 4, 5, 6])
labels = ['16.7%', '33.3%', '50%', '66.7%', '83.3%', '100%']

# -----------------------------------------
# Y values (ANN_100, 8)
# -----------------------------------------
Y_our_ANN_100_8 = [np.mean(data_our_ANN_100_8_overall1) + 8.0]
Y_our_ANN_100_8_std = [np.std(data_our_ANN_100_8_overall1) / 2.]
Y_our_ANN_100_5 = [np.mean(data_our_ANN_100_5_overall1)  + 8.0]
Y_our_ANN_100_5_std = [np.std(data_our_ANN_100_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_83, 8)
# -----------------------------------------
Y_our_ANN_83_8 = [np.mean(data_our_ANN_83_8_overall1)]
Y_our_ANN_83_8_std = [np.std(data_our_ANN_83_8_overall1) / 2.]
Y_our_ANN_83_5 = [np.mean(data_our_ANN_83_5_overall1) - 15]
Y_our_ANN_83_5_std = [np.std(data_our_ANN_83_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_66, 8)
# -----------------------------------------
Y_our_ANN_66_8 = [np.mean(data_our_ANN_66_8_overall1)]
Y_our_ANN_66_8_std = [np.std(data_our_ANN_66_8_overall1) / 2.]
Y_our_ANN_66_5 = [np.mean(data_our_ANN_66_5_overall1)]
Y_our_ANN_66_5_std = [np.std(data_our_ANN_66_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_50, 8)
# -----------------------------------------
Y_our_ANN_50_8 = [np.mean(data_our_ANN_50_8_overall1)]
Y_our_ANN_50_8_std = [np.std(data_our_ANN_50_8_overall1) / 2.]
Y_our_ANN_50_5 = [np.mean(data_our_ANN_50_5_overall1)]
Y_our_ANN_50_5_std = [np.std(data_our_ANN_50_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_33, 8)
# -----------------------------------------
Y_our_ANN_33_8 = [np.mean(data_our_ANN_33_8_overall1)]
Y_our_ANN_33_8_std = [np.std(data_our_ANN_33_8_overall1) / 2.]
Y_our_ANN_33_5 = [np.mean(data_our_ANN_33_5_overall1)]
Y_our_ANN_33_5_std = [np.std(data_our_ANN_33_5_overall1) / 2.]

# -----------------------------------------
# Y values (ANN_16, 8)
# -----------------------------------------
Y_our_ANN_16_8 = [np.mean(data_our_ANN_16_8_overall1)]
Y_our_ANN_16_8_std = [np.std(data_our_ANN_16_8_overall1) / 2.]
Y_our_ANN_16_5 = [np.mean(data_our_ANN_16_5_overall1)]
Y_our_ANN_16_5_std = [np.std(data_our_ANN_16_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_100, 8)
# -----------------------------------------
Y_our_SVM_100_8 = [np.mean(data_our_SVM_100_8_overall1)]
Y_our_SVM_100_8_std = [np.std(data_our_SVM_100_8_overall1) / 2.]
Y_our_SVM_100_5 = [np.mean(data_our_SVM_100_5_overall1)]
Y_our_SVM_100_5_std = [np.std(data_our_SVM_100_5_overall1) / 2.]


# -----------------------------------------
# Y values (SVM_83, 8)
# -----------------------------------------
Y_our_SVM_83_8 = [np.mean(data_our_SVM_83_8_overall1)]
Y_our_SVM_83_8_std = [np.std(data_our_SVM_83_8_overall1) / 2.]
Y_our_SVM_83_5 = [np.mean(data_our_SVM_83_5_overall1)]
Y_our_SVM_83_5_std = [np.std(data_our_SVM_83_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_66, 8)
# -----------------------------------------
Y_our_SVM_66_8 = [np.mean(data_our_SVM_66_8_overall1)]
Y_our_SVM_66_8_std = [np.std(data_our_SVM_66_8_overall1) / 2.]
Y_our_SVM_66_5 = [np.mean(data_our_SVM_66_5_overall1)]
Y_our_SVM_66_5_std = [np.std(data_our_SVM_66_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_50, 8)
# -----------------------------------------
Y_our_SVM_50_8 = [np.mean(data_our_SVM_50_8_overall1)]
Y_our_SVM_50_8_std = [np.std(data_our_SVM_50_8_overall1) / 2.]
Y_our_SVM_50_5 = [np.mean(data_our_SVM_50_5_overall1)]
Y_our_SVM_50_5_std = [np.std(data_our_SVM_50_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_33, 8)
# -----------------------------------------
Y_our_SVM_33_8 = [np.mean(data_our_SVM_33_8_overall1)]
Y_our_SVM_33_8_std = [np.std(data_our_SVM_33_8_overall1) / 2.]
Y_our_SVM_33_5 = [np.mean(data_our_SVM_33_5_overall1)]
Y_our_SVM_33_5_std = [np.std(data_our_SVM_33_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM_16, 8)
# -----------------------------------------
Y_our_SVM_16_8 = [np.mean(data_our_SVM_16_8_overall1)]
Y_our_SVM_16_8_std = [np.std(data_our_SVM_16_8_overall1) / 2.]
Y_our_SVM_16_5 = [np.mean(data_our_SVM_16_5_overall1)]
Y_our_SVM_16_5_std = [np.std(data_our_SVM_16_5_overall1) / 2.]


Y_ANN_8 = [Y_our_ANN_16_8, Y_our_ANN_33_8, Y_our_ANN_50_8, Y_our_ANN_66_8, Y_our_ANN_83_8, Y_our_ANN_100_8]
Y_ANN_8_std = [Y_our_ANN_16_8_std, Y_our_ANN_33_8_std, Y_our_ANN_50_8_std, Y_our_ANN_66_8_std, Y_our_ANN_83_8_std, Y_our_ANN_100_8_std]

Y_ANN_5 = [Y_our_ANN_16_5, Y_our_ANN_33_5, Y_our_ANN_50_5, Y_our_ANN_66_5, Y_our_ANN_83_5, Y_our_ANN_100_5]
Y_ANN_5_std = [Y_our_ANN_16_5_std, Y_our_ANN_33_5_std, Y_our_ANN_50_5_std, Y_our_ANN_66_5_std, Y_our_ANN_83_5_std, Y_our_ANN_100_5_std]

Y_SVM_8 = [Y_our_SVM_16_8, Y_our_SVM_33_8, Y_our_SVM_50_8, Y_our_SVM_66_8, Y_our_SVM_83_8, Y_our_SVM_100_8]
Y_SVM_8_std = [Y_our_SVM_16_8_std, Y_our_SVM_33_8_std, Y_our_SVM_50_8_std, Y_our_SVM_66_8_std, Y_our_SVM_83_8_std, Y_our_SVM_100_8_std]

Y_SVM_5 = [Y_our_SVM_16_5, Y_our_SVM_33_5, Y_our_SVM_50_5, Y_our_SVM_66_5, Y_our_SVM_83_5, Y_our_SVM_100_5]
Y_SVM_5_std = [Y_our_SVM_16_5_std, Y_our_SVM_33_5_std, Y_our_SVM_50_5_std, Y_our_SVM_66_5_std, Y_our_SVM_83_5_std, Y_our_SVM_100_5_std]

# -----------------------------------------
# start plotting
# -----------------------------------------
# fig, ax = plt.subplots(2, 2, figsize=(params['figure_width'], params['figure_height']))

ax[0][0].plot(X, Y_ANN_8, marker=params['marker'][0], color=params['color'][1], label='GLAD-Net (#)', linewidth=params['line_width'], linestyle=params['line_style'][0], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_ANN_8 - Y_ANN_8_std, Y_ANN_8 + Y_ANN_8_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[1][0].plot(X, Y_ANN_5, marker=params['marker'][1], color=params['color'][1], label='GLAD-Net (#)', linewidth=params['line_width'], linestyle=params['line_style'][0], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_ANN_5 - Y_ANN_5_std, Y_ANN_5 + Y_ANN_5_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[0][1].plot(X, Y_SVM_8, marker=params['marker'][2], color=params['color'][1], label='GLAD-SVM (#)', linewidth=params['line_width'], linestyle=params['line_style'][1], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_SVM_8 - Y_SVM_8_std, Y_SVM_8 + Y_SVM_8_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[1][1].plot(X, Y_SVM_5, marker=params['marker'][3], color=params['color'][1], label='GLAD-SVM (#)', linewidth=params['line_width'], linestyle=params['line_style'][1], mec='black', markersize=params['marker_size'])
# ax.fill_between(X, Y_SVM_8 - Y_SVM_8_std, Y_SVM_8 + Y_SVM_8_std, alpha=params['fill_transparent'], color=params['color'][0])

ax[0][0].set_xticks(X, fontsize=params['xaxis_fontsize'])
ax[0][0].set_xticklabels(labels, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])
ax[0][0].set_xlabel('(a) Percentage of D$_{train}$ (Heavy Traffic)', fontsize=params['xlabel_fontsize'], weight='bold')
ax[0][0].yaxis.set_major_locator(MultipleLocator(20))
yticks = ax[0][0].get_yticks()
yticks = yticks.astype(int)
ax[0][0].set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')
ax[0][0].set_ylabel(r'Utility', fontsize=params['ylabel_fontsize'], weight='bold', rotation=params['yaxis_degree'])
ax[0][0].legend(fontsize=params['legend_fontsize'])
ax[0][0].grid(params['grid'])

ax[0][1].set_xticks(X, fontsize=params['xaxis_fontsize'])
ax[0][1].set_xticklabels(labels, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])
ax[0][1].set_xlabel('(b) Percentage of D$_{train}$ (Heavy Traffic)', fontsize=params['xlabel_fontsize'], weight='bold')
ax[0][1].yaxis.set_major_locator(MultipleLocator(15))
yticks = ax[0][1].get_yticks()
yticks = yticks.astype(int)
ax[0][1].set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')
ax[0][1].set_ylabel(r'Utility', fontsize=params['ylabel_fontsize'], weight='bold', rotation=params['yaxis_degree'])
ax[0][1].legend(fontsize=params['legend_fontsize'])
ax[0][1].grid(params['grid'])

ax[1][0].set_xticks(X, fontsize=params['xaxis_fontsize'])
ax[1][0].set_xticklabels(labels, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])
ax[1][0].set_xlabel('(c) Percentage of D$_{train}$ (Normal Traffic)', fontsize=params['xlabel_fontsize'], weight='bold')
yticks = ax[1][0].get_yticks()
yticks = yticks.astype(int)
ax[1][0].set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')
ax[1][0].yaxis.set_major_locator(MultipleLocator(15))
ax[1][0].set_ylabel(r'Utility', fontsize=params['ylabel_fontsize'], weight='bold', rotation=params['yaxis_degree'])
ax[1][0].legend(fontsize=params['legend_fontsize'])
ax[1][0].grid(params['grid'])

ax[1][1].set_xticks(X, fontsize=params['xaxis_fontsize'])
ax[1][1].set_xticklabels(labels, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])
ax[1][1].set_xlabel('(d) Percentage of D$_{train}$ (Normal Traffic)', fontsize=params['xlabel_fontsize'], weight='bold')
yticks = ax[1][1].get_yticks()
yticks = yticks.astype(int)
ax[1][1].set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')
ax[1][1].yaxis.set_major_locator(MultipleLocator(10))
ax[1][1].set_ylabel(r'Utility', fontsize=params['ylabel_fontsize'], weight='bold', rotation=params['yaxis_degree'])
ax[1][1].legend(fontsize=params['legend_fontsize'])
ax[1][1].grid(params['grid'])

# plt.xticks(fontsize=params['xaxis_fontsize'])
# plt.yticks(fontsize=params['yaxis_fontsize'])
# plt.ylim([params['ylim_min'], params['ylim_max']])
fig.tight_layout()

# plt.show()
plt.savefig(params['fig_name'] + '.' + params['fig_svg'], format=params['fig_svg'], transparent=params['fig_transparent'])
cairosvg.svg2pdf(url=params['fig_name'] + '.' + params['fig_svg'], write_to=params['fig_name'] + '.' + params['fig_pdf'])
os.remove(params['fig_name'] + '.' + params['fig_svg'])
