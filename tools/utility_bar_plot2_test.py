#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot utility bar of Our method under different settings!
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

# -----------------------------------------
# styles
# -----------------------------------------
params = {
    # color
    'color': ['#FFFFCC', '#CCFFFF', '#FFCCCC'],
    'color2': ['#FFCCCC', '#FFFF99', '#CCCCFF'],
    'color3': ['#FFCC99', '#CCFF99', '#CCCCCC'],
    'color4': ['#FF9966', '#FF6666', '#FFCCCC'],
    'color5': ['#FFCCCC', '#FFCCCC', '#CCFFFF'],

    'ecolor': ['black'],
    
    'marker': ['v', '*', 'o', '^', '<', '>'], # refer to https://matplotlib.org/stable/api/markers_api.html
    'marker_size': 12,
    
    'bar_width': 0.3,
    'bar_interval': 0.2,
    
    'error_capsize': 5,
    
    'xaxis_fontsize': 12,
    'xaxis_degree': 0,
    'xaxis_fontweight': 'normal',
    'xlabel_fontsize': 18,
    'xlabel_fontweight': 'normal',
    'xlim_min': 1,
    'xlim_max': 10,

    'yaxis_fontsize': 12,
    'yaxis_degree': 90,
    'yaxis_fontweight': 'normal',
    'ylabel_fontsize': 18,
    'ylabel_fontweight': 'normal',
    'ylim_min': 70,
    'ylim_max': 92,

    'legend_fontsize': 18,
    
    'figure_weight': 6,
    'figure_height': 4,
    
    'grid': True,

    'fill_transparent': 0.5,

    'line_width': 2,

    'hatch': ['\\\\', 'o', '//', '.', '-', 'x', 'o', 'O', '+', '*'],
    
    'fig_name': 'fig_utility',
    'fig_svg': 'svg',
    'fig_png': 'png',
    'fig_jpeg': 'jpeg',
    'fig_pdf': 'pdf',
    'fig_transparent': False,
}
params_eva = {
    'threshold_cost': 5000, # if the task is completed within time threshold
    'penalty_cost': 5000, # cost penalty
    'penalty_risk': 15000, # risk penalty
    'constant': 8000,
}
params_debug = {
    'print': False
}

# -----------------------------------------
# data processing: Our (ANN, 8)
# -----------------------------------------
path_our_ANN_8 = utils.root_path() + 'results/8/test_ANN_8/'
data_our_ANN_8_overall1, data_our_ANN_8_overall2, data_our_ANN_8_cost, data_our_ANN_8_pref, data_our_ANN_8_brisk, data_our_ANN_8_risk, data_our_ANN_8_avoid, data_our_ANN_8_collision = utils.read_result(path_our_ANN_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
if params_debug['print']:
    print('-'*30)
    print('row of data_our_ANN_8_overall1:{}'.format((data_our_ANN_8_overall1)))
print('-'*30)
print('data_our_ANN_8_overall1:{}'.format(np.mean(data_our_ANN_8_overall1)))
print('data_our_ANN_8_cost:{}'.format(np.mean(data_our_ANN_8_cost)))
print('data_our_ANN_8_pref:{}'.format(np.mean(data_our_ANN_8_pref)))
print('data_our_ANN_8_brisk:{}'.format(np.mean(data_our_ANN_8_brisk)))
print('data_our_ANN_8_risk:{}'.format(np.mean(data_our_ANN_8_risk)))
print('data_our_ANN_8_avoid:{}'.format(np.mean(data_our_ANN_8_avoid)))
print('data_our_ANN_8_collision:{}'.format(np.mean(data_our_ANN_8_collision)))

# path_our_ANN_70_8 = utils.root_path() + 'results/8/test_ANN_70_8/'
# data_our_ANN_70_8_overall1, data_our_ANN_70_8_overall2, data_our_ANN_70_8_cost, data_our_ANN_70_8_pref, data_our_ANN_70_8_brisk, data_our_ANN_70_8_risk, data_our_ANN_70_8_avoid, data_our_ANN_70_8_collision = utils.read_result(path_our_ANN_70_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
# if params_debug['print']:
#     print('-'*30)
#     print('row of data_our_ANN_70_8_overall1:{}'.format((data_our_ANN_70_8_overall1)))
# print('-'*30)
# print('data_our_ANN_70_8_overall1:{}'.format(np.mean(data_our_ANN_70_8_overall1)))
# print('data_our_ANN_70_8_cost:{}'.format(np.mean(data_our_ANN_70_8_cost)))
# print('data_our_ANN_70_8_pref:{}'.format(np.mean(data_our_ANN_70_8_pref)))
# print('data_our_ANN_70_8_brisk:{}'.format(np.mean(data_our_ANN_70_8_brisk)))
# print('data_our_ANN_70_8_risk:{}'.format(np.mean(data_our_ANN_70_8_risk)))
# print('data_our_ANN_70_8_avoid:{}'.format(np.mean(data_our_ANN_70_8_avoid)))
# print('data_our_ANN_70_8_collision:{}'.format(np.mean(data_our_ANN_70_8_collision)))

# path_our_ANN_40_8 = utils.root_path() + 'results/8/test_ANN_40_8/'
# data_our_ANN_40_8_overall1, data_our_ANN_40_8_overall2, data_our_ANN_40_8_cost, data_our_ANN_40_8_pref, data_our_ANN_40_8_brisk, data_our_ANN_40_8_risk, data_our_ANN_40_8_avoid, data_our_ANN_40_8_collision = utils.read_result(path_our_ANN_40_8, params_eva['threshold_cost'], params_eva['penalty_cost'], params_eva['penalty_risk'], params_eva['constant'])
# if params_debug['print']:
#     print('-'*30)
#     print('row of data_our_ANN_40_8_overall1:{}'.format((data_our_ANN_40_8_overall1)))
# print('-'*30)
# print('data_our_ANN_40_8_overall1:{}'.format(np.mean(data_our_ANN_40_8_overall1)))
# print('data_our_ANN_40_8_cost:{}'.format(np.mean(data_our_ANN_40_8_cost)))
# print('data_our_ANN_40_8_pref:{}'.format(np.mean(data_our_ANN_40_8_pref)))
# print('data_our_ANN_40_8_brisk:{}'.format(np.mean(data_our_ANN_40_8_brisk)))
# print('data_our_ANN_40_8_risk:{}'.format(np.mean(data_our_ANN_40_8_risk)))
# print('data_our_ANN_40_8_avoid:{}'.format(np.mean(data_our_ANN_40_8_avoid)))
# print('data_our_ANN_40_8_collision:{}'.format(np.mean(data_our_ANN_40_8_collision)))


# -----------------------------------------
# Y values (ANN, 8)
# -----------------------------------------
Y_our_ANN_8 = [np.mean(data_our_ANN_8_overall1)]
Y_our_ANN_8_std = [np.std(data_our_ANN_8_overall1)]

Y_our_ANN_70_8 = [np.mean(data_our_ANN_70_8_overall1)]
Y_our_ANN_70_8_std = [np.std(data_our_ANN_70_8_overall1)]

Y_our_ANN_40_8 = [np.mean(data_our_ANN_40_8_overall1)]
Y_our_ANN_40_8_std = [np.std(data_our_ANN_40_8_overall1)]


# -----------------------------------------
# X values
# -----------------------------------------
X = [1, 2, 3]
xticks = ['GLAD-KNN', 'GLAD-XGBC', 'GLAD-ANN']

# -----------------------------------------
# plotting
# -----------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(params['figure_weight'], params['figure_height']))

# -----------------------------------------
# plot our (ANN)
# -----------------------------------------
ax.bar(X[1] - params['bar_interval'], Y_our_ANN_40_8, width=params['bar_width'], yerr=Y_our_ANN_40_8_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])
ax.bar(X[1], Y_our_ANN_70_8, width=params['bar_width'], yerr=Y_our_ANN_70_8_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])
ax.bar(X[1] + params['bar_interval'], Y_our_ANN_8, width=params['bar_width'], yerr=Y_our_ANN_8_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])

# -----------------------------------------
# xlabel
# -----------------------------------------
ax.set_xlabel('Approach', fontsize=params['xlabel_fontsize'], fontweight=params['xlabel_fontweight'], rotation=params['xaxis_degree'])
ax.set_xticks(X[:3])
ax.set_xticklabels(xticks, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])

# -----------------------------------------
# ylabel
# -----------------------------------------
ax.yaxis.set_major_locator(ticker.MultipleLocator(2000))
ax.set_ylabel('Utility', fontsize=params['ylabel_fontsize'], weight=params['ylabel_fontweight'], rotation=params['yaxis_degree'])
ax.tick_params(axis='y', labelsize=params['yaxis_fontsize'], rotation=params['yaxis_degree'])

# -----------------------------------------
# legend
# -----------------------------------------
# ax.legend([legend1, legend2], [r'Heavy Traffic ($\lambda = 0.1$)', r'Normal Traffic ($\lambda = 0.08$)'])

# -----------------------------------------
# styles
# -----------------------------------------
fig.tight_layout()
ax.grid(params['grid'])

# -----------------------------------------
# show and save figure
# -----------------------------------------
plt.show()
# plt.savefig(params['fig_name'] + '.' + params['fig_svg'], format=params['fig_svg'], transparent=params['fig_transparent'])
# cairosvg.svg2pdf(url=params['fig_name'] + '.' + params['fig_svg'], write_to=params['fig_name'] + '.' + params['fig_pdf'])
# os.remove(params['fig_name'] + '.' + params['fig_svg'])