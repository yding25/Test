#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot scatter!
# -----------------------------------------

from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    'color': ['#2878B5', '#9AC9DB', '#F3D266', '#C82423', '#FF8884'],

    'ecolor': ['black'],
    
    'marker': ['d', '*', 'o', 's', 's', '>'], # refer to https://matplotlib.org/stable/api/markers_api.html
    'marker_size': 12,
    
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

    'legend_fontsize': 16,
    'legend_location': ['best', 'upper right'],
    
    'figure_width': 12,
    'figure_height': 4,
    
    'grid': True,

    'fill_transparent': 0.5,

    'line_width': 2,

    'size': np.array([300, 300, 300, 300, 300]),

    'hatch': ['\\\\', 'o', '//', '.', '-', 'x', 'o', 'O', '+', '*'],
    
    'fig_name': 'fig_2D',
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
    'constant': 8000,
}
params_debug = {
    'print': False
}


# -----------------------------------------
# data processing: Our (ANN, 8)
# -----------------------------------------
path_our_ANN_8 = utils.root_path() + 'results/8/test_ANN_100_8_nonew/'
data_our_ANN_8_overall1, data_our_ANN_8_overall2, data_our_ANN_8_cost, data_our_ANN_8_pref, data_our_ANN_8_brisk, data_our_ANN_8_collision, data_our_ANN_8_risk, data_our_ANN_8_avoid = utils.read_result(path_our_ANN_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_8_overall1:{}'.format((data_our_ANN_8_overall1)))
print('-'*30)
print('data_our_ANN_8_overall1:{}'.format(np.mean(data_our_ANN_8_overall1)))
print('data_our_ANN_8_cost:{}'.format(np.mean(data_our_ANN_8_cost)))
print('data_our_ANN_8_pref:{}'.format(np.mean(data_our_ANN_8_pref)))
print('data_our_ANN_8_brisk:{}'.format(np.mean(data_our_ANN_8_brisk)))
print('data_our_ANN_8_collision:{}'.format(np.mean(data_our_ANN_8_collision)))
print('data_our_ANN_8_risk:{}'.format(np.mean(data_our_ANN_8_risk)))
print('data_our_ANN_8_avoid:{}'.format(np.mean(data_our_ANN_8_avoid)))

# -----------------------------------------
# data processing: Our (ANN, 5)
# -----------------------------------------
path_our_ANN_5 = utils.root_path() + 'results/5/test_ANN_100_5_nonew/'
data_our_ANN_5_overall1, data_our_ANN_5_overall2, data_our_ANN_5_cost, data_our_ANN_5_pref, data_our_ANN_5_brisk, data_our_ANN_5_collision, data_our_ANN_5_risk, data_our_ANN_5_avoid = utils.read_result(path_our_ANN_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_5_overall1:{}'.format((data_our_ANN_5_overall1)))
print('-'*30)
print('data_our_ANN_5_overall1:{}'.format(np.mean(data_our_ANN_5_overall1)))
print('data_our_ANN_5_cost:{}'.format(np.mean(data_our_ANN_5_cost)))
print('data_our_ANN_5_pref:{}'.format(np.mean(data_our_ANN_5_pref)))
print('data_our_ANN_5_brisk:{}'.format(np.mean(data_our_ANN_5_brisk)))
print('data_our_ANN_5_collision:{}'.format(np.mean(data_our_ANN_5_collision)))
print('data_our_ANN_5_risk:{}'.format(np.mean(data_our_ANN_5_risk)))
print('data_our_ANN_5_avoid:{}'.format(np.mean(data_our_ANN_5_avoid)))


# -----------------------------------------
# data processing: Our (SVM, 8)
# -----------------------------------------
path_our_SVM_8 = utils.root_path() + 'results/8/test_SVM_100_8_nonew/'
data_our_SVM_8_overall1, data_our_SVM_8_overall2, data_our_SVM_8_cost, data_our_SVM_8_pref, data_our_SVM_8_brisk, data_our_SVM_8_collision, data_our_SVM_8_risk, data_our_SVM_8_avoid = utils.read_result(path_our_SVM_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_8_overall1:{}'.format((data_our_SVM_8_overall1)))
print('-'*30)
print('data_our_SVM_8_overall1:{}'.format(np.mean(data_our_SVM_8_overall1)))
print('data_our_SVM_8_cost:{}'.format(np.mean(data_our_SVM_8_cost)))
print('data_our_SVM_8_pref:{}'.format(np.mean(data_our_SVM_8_pref)))
print('data_our_SVM_8_brisk:{}'.format(np.mean(data_our_SVM_8_brisk)))
print('data_our_SVM_8_brisk:{}'.format(np.mean(data_our_SVM_8_collision)))
print('data_our_SVM_8_risk:{}'.format(np.mean(data_our_SVM_8_risk)))
print('data_our_SVM_8_brisk:{}'.format(np.mean(data_our_SVM_8_avoid)))

# -----------------------------------------
# data processing: Our (SVM, 5)
# -----------------------------------------
path_our_SVM_5 = utils.root_path() + 'results/5/test_SVM_100_5_nonew/'
data_our_SVM_5_overall1, data_our_SVM_5_overall2, data_our_SVM_5_cost, data_our_SVM_5_pref, data_our_SVM_5_brisk, data_our_SVM_5_collision, data_our_SVM_5_risk, data_our_SVM_5_avoid = utils.read_result(path_our_SVM_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_SVM_5_overall1:{}'.format((data_our_SVM_5_overall1)))
print('-'*30)
print('data_our_SVM_5_overall1:{}'.format(np.mean(data_our_SVM_5_overall1)))
print('data_our_SVM_5_cost:{}'.format(np.mean(data_our_SVM_5_cost)))
print('data_our_SVM_5_pref:{}'.format(np.mean(data_our_SVM_5_pref)))
print('data_our_SVM_5_brisk:{}'.format(np.mean(data_our_SVM_5_brisk)))
print('data_our_SVM_5_brisk:{}'.format(np.mean(data_our_SVM_5_collision)))
print('data_our_SVM_5_risk:{}'.format(np.mean(data_our_SVM_5_risk)))
print('data_our_SVM_5_brisk:{}'.format(np.mean(data_our_SVM_5_avoid)))

# -----------------------------------------
# data processing: nosafe (ANN, 8)
# -----------------------------------------
path_nosafe_8 = utils.root_path() + 'results/8/nosafe_ANN_8/'
data_nosafe_8_overall1, data_nosafe_8_overall2, data_nosafe_8_cost, data_nosafe_8_pref, data_nosafe_8_brisk, data_nosafe_8_collision, data_nosafe_8_risk, data_nosafe_8_avoid = utils.read_result(path_nosafe_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_nosafe_8_overall1:{}'.format((data_nosafe_8_overall1)))
print('-'*30)
print('data_nosafe_8_overall1:{}'.format(np.mean(data_nosafe_8_overall1)))
print('data_nosafe_8_cost:{}'.format(np.mean(data_nosafe_8_cost)))
print('data_nosafe_8_pref:{}'.format(np.mean(data_nosafe_8_pref)))
print('data_nosafe_8_brisk:{}'.format(np.mean(data_nosafe_8_brisk)))
print('data_nosafe_8_collision:{}'.format(np.mean(data_nosafe_8_collision)))
print('data_nosafe_8_risk:{}'.format(np.mean(data_nosafe_8_risk)))
print('data_nosafe_8_avoid:{}'.format(np.mean(data_nosafe_8_avoid)))

# -----------------------------------------
# data processing: nosafe (ANN, 5)
# -----------------------------------------
path_nosafe_5 = utils.root_path() + 'results/5/nosafe_ANN_5/'
data_nosafe_5_overall1, data_nosafe_5_overall2, data_nosafe_5_cost, data_nosafe_5_pref, data_nosafe_5_brisk, data_nosafe_5_collision, data_nosafe_5_risk, data_nosafe_5_avoid = utils.read_result(path_nosafe_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_nosafe_5_overall1:{}'.format((data_nosafe_5_overall1)))
print('-'*30)
print('data_nosafe_5_overall1:{}'.format(np.mean(data_nosafe_5_overall1)))
print('data_nosafe_5_cost:{}'.format(np.mean(data_nosafe_5_cost)))
print('data_nosafe_5_pref:{}'.format(np.mean(data_nosafe_5_pref)))
print('data_nosafe_5_brisk:{}'.format(np.mean(data_nosafe_5_brisk)))
print('data_nosafe_5_collision:{}'.format(np.mean(data_nosafe_5_collision)))
print('data_nosafe_5_risk:{}'.format(np.mean(data_nosafe_5_risk)))
print('data_nosafe_5_avoid:{}'.format(np.mean(data_nosafe_5_avoid)))


# -----------------------------------------
# data processing: nocost (ANN, 8)
# -----------------------------------------
path_nocost_8 = utils.root_path() + 'results/8/nocost_ANN_8/'
data_nocost_8_overall1, data_nocost_8_overall2, data_nocost_8_cost, data_nocost_8_pref, data_nocost_8_brisk, data_nocost_8_collision, data_nocost_8_risk, data_nocost_8_avoid = utils.read_result(path_nocost_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_nocost_8_overall1:{}'.format((data_nocost_8_overall1)))
print('-'*30)
print('data_nocost_8_overall1:{}'.format(np.mean(data_nocost_8_overall1)))
print('data_nocost_8_cost:{}'.format(np.mean(data_nocost_8_cost)))
print('data_nocost_8_pref:{}'.format(np.mean(data_nocost_8_pref)))
print('data_nocost_8_brisk:{}'.format(np.mean(data_nocost_8_brisk)))
print('data_nocost_8_collision:{}'.format(np.mean(data_nocost_8_collision)))
print('data_nocost_8_risk:{}'.format(np.mean(data_nocost_8_risk)))
print('data_nocost_8_avoid:{}'.format(np.mean(data_nocost_8_avoid)))

# -----------------------------------------
# data processing: nocost (ANN, 5)
# -----------------------------------------
path_nocost_5 = utils.root_path() + 'results/5/nocost_ANN_5/'
data_nocost_5_overall1, data_nocost_5_overall2, data_nocost_5_cost, data_nocost_5_pref, data_nocost_5_brisk, data_nocost_5_collision, data_nocost_5_risk, data_nocost_5_avoid = utils.read_result(path_nocost_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_nocost_5_overall1:{}'.format((data_nocost_5_overall1)))
print('-'*30)
print('data_nocost_5_overall1:{}'.format(np.mean(data_nocost_5_overall1)))
print('data_nocost_5_cost:{}'.format(np.mean(data_nocost_5_cost)))
print('data_nocost_5_pref:{}'.format(np.mean(data_nocost_5_pref)))
print('data_nocost_5_brisk:{}'.format(np.mean(data_nocost_5_brisk)))
print('data_nocost_5_collision:{}'.format(np.mean(data_nocost_5_collision)))
print('data_nocost_5_risk:{}'.format(np.mean(data_nocost_5_risk)))
print('data_nocost_5_avoid:{}'.format(np.mean(data_nocost_5_avoid)))

# -----------------------------------------
# data processing: nopref (ANN, 8)
# -----------------------------------------
path_nopref_8 = utils.root_path() + 'results/8/nopref_ANN_8/'
data_nopref_8_overall1, data_nopref_8_overall2, data_nopref_8_cost, data_nopref_8_pref, data_nopref_8_brisk, data_nopref_8_collision, data_nopref_8_risk, data_nopref_8_avoid = utils.read_result(path_nopref_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_nopref_8_overall:{}'.format((data_nopref_8_overall1)))
print('-'*30)
print('data_nopref_8_overall1:{}'.format(np.mean(data_nopref_8_overall1)))
print('data_nopref_8_cost:{}'.format(np.mean(data_nopref_8_cost)))
print('data_nopref_8_pref:{}'.format(np.mean(data_nopref_8_pref)))
print('data_nopref_8_brisk:{}'.format(np.mean(data_nopref_8_brisk)))
print('data_nopref_8_collision:{}'.format(np.mean(data_nopref_8_collision)))
print('data_nopref_8_risk:{}'.format(np.mean(data_nopref_8_risk)))
print('data_nopref_8_avoid:{}'.format(np.mean(data_nopref_8_avoid)))

# -----------------------------------------
# data processing: nopref (ANN, 5)
# -----------------------------------------
path_nopref_5 = utils.root_path() + 'results/5/nopref_ANN_5/'
data_nopref_5_overall1, data_nopref_5_overall2, data_nopref_5_cost, data_nopref_5_pref, data_nopref_5_brisk, data_nopref_5_collision, data_nopref_5_risk, data_nopref_5_avoid = utils.read_result(path_nopref_5, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_nopref_5_overall:{}'.format((data_nopref_5_overall1)))
print('-'*30)
print('data_nopref_5_overall1:{}'.format(np.mean(data_nopref_5_overall1)))
print('data_nopref_5_cost:{}'.format(np.mean(data_nopref_5_cost)))
print('data_nopref_5_pref:{}'.format(np.mean(data_nopref_5_pref)))
print('data_nopref_5_brisk:{}'.format(np.mean(data_nopref_5_brisk)))
print('data_nopref_5_collision:{}'.format(np.mean(data_nopref_5_collision)))
print('data_nopref_5_risk:{}'.format(np.mean(data_nopref_5_risk)))
print('data_nopref_5_avoid:{}'.format(np.mean(data_nopref_5_avoid)))


# -----------------------------------------
# Y values (ANN, 8)
# -----------------------------------------
Y_our_ANN_8 = [np.mean(data_our_ANN_8_overall1)]
Y_our_ANN_8_std = [np.std(data_our_ANN_8_overall1) / 2.]

# -----------------------------------------
# Y values (ANN, 5)
# -----------------------------------------
Y_our_ANN_5 = [np.mean(data_our_ANN_5_overall1)]
Y_our_ANN_5_std = [np.std(data_our_ANN_5_overall1) / 2.]

# -----------------------------------------
# Y values (SVM, 8)
# -----------------------------------------
Y_our_SVM_8 = [np.mean(data_our_SVM_8_overall1)]
Y_our_SVM_8_std = [np.std(data_our_SVM_8_overall1) / 2.]

# -----------------------------------------
# Y values (SVM, 5)
# -----------------------------------------
Y_our_SVM_5 = [np.mean(data_our_SVM_5_overall1)]
Y_our_SVM_5_std = [np.std(data_our_SVM_5_overall1) / 2.]

# -----------------------------------------
# Y values (NoSafe, 8)
# -----------------------------------------
Y_nosafe_8 = [np.mean(data_nosafe_8_overall1)]
Y_nosafe_8_std = [np.std(data_nosafe_8_overall1) / 2.]

# -----------------------------------------
# Y values (NoSafe, 5)
# -----------------------------------------
Y_nosafe_5 = [np.mean(data_nosafe_5_overall1)]
Y_nosafe_5_std = [np.std(data_nosafe_5_overall1) / 2.]

# -----------------------------------------
# Y values (NoCost, 8)
# -----------------------------------------
Y_nocost_8 = [np.mean(data_nocost_8_overall1)]
Y_nocost_8_std = [np.std(data_nocost_8_overall1) / 2.]

# -----------------------------------------
# Y values (NoCost, 5)
# -----------------------------------------
Y_nocost_5 = [np.mean(data_nocost_5_overall1)]
Y_nocost_5_std = [np.std(data_nocost_5_overall1) / 2.]

# -----------------------------------------
# Y values (NoPref, 8)
# -----------------------------------------
Y_nopref_8 = [np.mean(data_nopref_8_overall1)]
Y_nopref_8_std = [np.std(data_nopref_8_overall1) / 2.]

# -----------------------------------------
# Y values (NoPref, 5)
# -----------------------------------------
Y_nopref_5 = [np.mean(data_nopref_5_overall1)]
Y_nopref_5_std = [np.std(data_nopref_5_overall1) / 2.]

# -----------------------------------------
# plotting
# -----------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(params['figure_width'], params['figure_height']))


# figure 1
x1 = np.array([np.mean(data_nocost_8_cost), np.mean(data_nosafe_8_cost), np.mean(data_nopref_8_cost), np.mean(data_our_SVM_8_cost), np.mean(data_our_ANN_8_cost)])
y1 = np.array([np.mean(data_nocost_8_risk) * params_eva['penalty_risk'], np.mean(data_nosafe_8_risk) * params_eva['penalty_risk'], np.mean(data_nopref_8_risk) * params_eva['penalty_risk'], np.mean(data_our_ANN_8_risk) * params_eva['penalty_risk'], np.mean(data_our_SVM_8_risk) * params_eva['penalty_risk']])
ax1 = plt.subplot(1, 2, 1)
s11 = ax1.scatter(x1[0], y1[0], s=params['size'][0], c=params['color'][0], marker=params['marker'][0])
s12 = ax1.scatter(x1[1], y1[1], s=params['size'][1], c=params['color'][1], marker=params['marker'][1])
s13 = ax1.scatter(x1[2], y1[2], s=params['size'][2], c=params['color'][2], marker=params['marker'][2])
s14 = ax1.scatter(x1[3], y1[3], s=params['size'][3], c=params['color'][3], marker=params['marker'][3])
s15 = ax1.scatter(x1[4], y1[4], s=params['size'][4], c=params['color'][4], marker=params['marker'][4])
ax1.set_xlabel('Cost', fontsize=params['xlabel_fontsize'], weight=params['xlabel_fontweight'], rotation=params['xaxis_degree'])
ax1.set_ylabel('Unsafe', fontsize=params['ylabel_fontsize'], weight=params['ylabel_fontweight'], rotation=params['yaxis_degree'])
ax1.legend((s11, s12, s13, s14, s15), ('MIND', 'TPAD', 'TMPUD', 'GLAD-Net (ours)', 'GLAD-SVM (ours)'), loc=params['legend_location'][0], scatteryoffsets=[0.1, 0.1, 0.1, 0.1, 0.1], fontsize=params['legend_fontsize'])
ax1.set_xlim([-500, 6000])
ax1.set_ylim([-500, 6000])

yticks = ax1.get_yticks()
yticks = yticks.astype(int)
ax1.set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')
ax1.yaxis.set_major_locator(MultipleLocator(1500))
xticks = ax1.get_xticks()
xticks = xticks.astype(int)
ax1.set_xticklabels(xticks, fontsize=params['xaxis_fontsize'], fontweight=params['xaxis_fontweight'], rotation=params['xaxis_degree'], verticalalignment='center')
ax1.xaxis.set_major_locator(MultipleLocator(1500))

# figure 2
x1 = np.array([np.mean(data_nocost_8_pref), np.mean(data_nosafe_8_pref), np.mean(data_nopref_8_pref), np.mean(data_our_ANN_8_pref), np.mean(data_our_SVM_8_pref)])
y1 = np.array([np.mean(data_nocost_8_risk) * params_eva['penalty_risk'], np.mean(data_nosafe_8_risk) * params_eva['penalty_risk'], np.mean(data_nopref_8_risk) * params_eva['penalty_risk'], np.mean(data_our_ANN_8_risk) * params_eva['penalty_risk'], np.mean(data_our_SVM_8_risk) * params_eva['penalty_risk']])
ax2 = plt.subplot(1, 2, 2)
s11 = ax2.scatter(x1[0], y1[0], s=params['size'][0], c=params['color'][0], marker=params['marker'][0])
s12 = ax2.scatter(x1[1], y1[1], s=params['size'][1], c=params['color'][1], marker=params['marker'][1])
s13 = ax2.scatter(x1[2], y1[2], s=params['size'][2], c=params['color'][2], marker=params['marker'][2])
s14 = ax2.scatter(x1[3], y1[3], s=params['size'][3], c=params['color'][3], marker=params['marker'][3])
s15 = ax2.scatter(x1[4], y1[4], s=params['size'][4], c=params['color'][4], marker=params['marker'][4])
ax2.set_xlabel('Pref', fontsize=params['xlabel_fontsize'], weight=params['xlabel_fontweight'], rotation=params['xaxis_degree'])
ax2.set_ylabel('Unsafe', fontsize=params['ylabel_fontsize'], weight=params['ylabel_fontweight'], rotation=params['yaxis_degree'])
ax2.legend((s11, s12, s13, s14, s15), ('MIND', 'TPAD', 'TMPUD', 'GLAD-Net (ours)', 'GLAD-SVM (ours)'), loc=params['legend_location'][0], scatteryoffsets=[0.1, 0.1, 0.1, 0.1, 0.1], fontsize=params['legend_fontsize'])
ax2.set_xlim([-500, 6000])
ax2.set_ylim([-500, 6000])

yticks = ax2.get_yticks()
yticks = yticks.astype(int)
ax2.set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')
ax2.yaxis.set_major_locator(MultipleLocator(1500))
xticks = ax2.get_xticks()
xticks = xticks.astype(int)
ax2.set_xticklabels(xticks, fontsize=params['xaxis_fontsize'], fontweight=params['xaxis_fontweight'], rotation=params['xaxis_degree'], verticalalignment='center')
ax2.xaxis.set_major_locator(MultipleLocator(1500))

# figure 3
# x1 = np.array([np.mean(data_nocost_8_cost), np.mean(data_nosafe_8_cost), np.mean(data_nopref_8_cost), np.mean(data_our_ANN_8_cost), np.mean(data_our_SVM_8_cost)])
# y1 = np.array([np.mean(data_nocost_8_pref), np.mean(data_nosafe_8_pref), np.mean(data_nopref_8_pref), np.mean(data_our_ANN_8_pref), np.mean(data_our_SVM_8_pref)])
# ax3 = plt.subplot(1, 3, 3)
# s11 = ax3.scatter(x1[0], y1[0], s=params['size'][0], c=params['color'][0], marker=params['marker'][0])
# s12 = ax3.scatter(x1[1], y1[1], s=params['size'][1], c=params['color'][1], marker=params['marker'][1])
# s13 = ax3.scatter(x1[2], y1[2], s=params['size'][2], c=params['color'][2], marker=params['marker'][2])
# s14 = ax3.scatter(x1[3], y1[3], s=params['size'][3], c=params['color'][3], marker=params['marker'][3])
# s15 = ax3.scatter(x1[4], y1[4], s=params['size'][4], c=params['color'][4], marker=params['marker'][4])
# ax3.set_xlabel('Cost', fontsize=params['xlabel_fontsize'], weight=params['xlabel_fontweight'], rotation=params['xaxis_degree'])
# ax3.set_ylabel('Preference', fontsize=params['ylabel_fontsize'], weight=params['ylabel_fontweight'], rotation=params['yaxis_degree'])
# ax3.legend((s11, s12, s13, s14, s15), ('MIND', 'TPAD', 'TMPUD', 'GLAD-Net (ours)', 'GLAD-SVM (ours)'), loc=params['legend_location'][0], scatteryoffsets=[0.1, 0.1, 0.1, 0.1, 0.1])
# ax3.set_xlim([-500, 3000])
# ax3.set_ylim([-500, 3000])

# -----------------------------------------
# styles
# -----------------------------------------
fig.tight_layout()
ax1.grid(params['grid'])
ax2.grid(params['grid'])
# ax3.grid(params['grid'])
# -----------------------------------------
# show and save figure
# -----------------------------------------
# plt.show()
plt.savefig(params['fig_name'] + '.' + params['fig_svg'], format=params['fig_svg'], transparent=params['fig_transparent'])
cairosvg.svg2pdf(url=params['fig_name'] + '.' + params['fig_svg'], write_to=params['fig_name'] + '.' + params['fig_pdf'])
os.remove(params['fig_name'] + '.' + params['fig_svg'])