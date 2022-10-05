#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot utility bar of four methods!
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
    'color': ['#9AC9DB', '#F8AC8C', '#2878B5'],

    'ecolor': ['black'],
    
    'marker': ['v', '*', 'o', '^', '<', '>'], # refer to https://matplotlib.org/stable/api/markers_api.html
    'marker_size': 12,
    
    'bar_width': 0.5,
    'bar_interval': 0.,
    
    'error_capsize': 5,
    
    'xaxis_fontsize': 12,
    'xaxis_degree': 35,
    'xaxis_fontweight': 'normal',
    'xlabel_fontsize': 18,
    'xlabel_fontweight': 'bold',
    'xlim_min': 1,
    'xlim_max': 10,

    'yaxis_fontsize': 12,
    'yaxis_degree': 90,
    'yaxis_fontweight': 'normal',
    'ylabel_fontsize': 18,
    'ylabel_fontweight': 'bold',
    'ylim_min': 70,
    'ylim_max': 92,

    'legend_fontsize': 16,
    
    'figure_width': 8,
    'figure_height': 6,
    
    'grid': True,

    'fill_transparent': 0.5,

    'line_width': 2,

    'hatch': ['\\\\', '//', '-', 'O', '.', 'x', 'o', '+', '*'],
    
    'fig_name': 'fig_utility',
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
    'constant': 7000,
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
# print('-'*30)
# print('data_our_ANN_5_overall1:{}'.format(np.mean(data_our_ANN_5_overall1)))
# print('data_our_ANN_5_cost:{}'.format(np.mean(data_our_ANN_5_cost)))
# print('data_our_ANN_5_pref:{}'.format(np.mean(data_our_ANN_5_pref)))
# print('data_our_ANN_5_brisk:{}'.format(np.mean(data_our_ANN_5_brisk)))
# print('data_our_ANN_5_collision:{}'.format(np.mean(data_our_ANN_5_collision)))
# print('data_our_ANN_5_risk:{}'.format(np.mean(data_our_ANN_5_risk)))
# print('data_our_ANN_5_avoid:{}'.format(np.mean(data_our_ANN_5_avoid)))


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
# print('-'*30)
# print('data_our_SVM_5_overall1:{}'.format(np.mean(data_our_SVM_5_overall1)))
# print('data_our_SVM_5_cost:{}'.format(np.mean(data_our_SVM_5_cost)))
# print('data_our_SVM_5_pref:{}'.format(np.mean(data_our_SVM_5_pref)))
# print('data_our_SVM_5_brisk:{}'.format(np.mean(data_our_SVM_5_brisk)))
# print('data_our_SVM_5_brisk:{}'.format(np.mean(data_our_SVM_5_collision)))
# print('data_our_SVM_5_risk:{}'.format(np.mean(data_our_SVM_5_risk)))
# print('data_our_SVM_5_brisk:{}'.format(np.mean(data_our_SVM_5_avoid)))

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
# print('-'*30)
# print('data_nosafe_5_overall1:{}'.format(np.mean(data_nosafe_5_overall1)))
# print('data_nosafe_5_cost:{}'.format(np.mean(data_nosafe_5_cost)))
# print('data_nosafe_5_pref:{}'.format(np.mean(data_nosafe_5_pref)))
# print('data_nosafe_5_brisk:{}'.format(np.mean(data_nosafe_5_brisk)))
# print('data_nosafe_5_collision:{}'.format(np.mean(data_nosafe_5_collision)))
# print('data_nosafe_5_risk:{}'.format(np.mean(data_nosafe_5_risk)))
# print('data_nosafe_5_avoid:{}'.format(np.mean(data_nosafe_5_avoid)))


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
# print('-'*30)
# print('data_nocost_5_overall1:{}'.format(np.mean(data_nocost_5_overall1)))
# print('data_nocost_5_cost:{}'.format(np.mean(data_nocost_5_cost)))
# print('data_nocost_5_pref:{}'.format(np.mean(data_nocost_5_pref)))
# print('data_nocost_5_brisk:{}'.format(np.mean(data_nocost_5_brisk)))
# print('data_nocost_5_collision:{}'.format(np.mean(data_nocost_5_collision)))
# print('data_nocost_5_risk:{}'.format(np.mean(data_nocost_5_risk)))
# print('data_nocost_5_avoid:{}'.format(np.mean(data_nocost_5_avoid)))

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
# print('-'*30)
# print('data_nopref_5_overall1:{}'.format(np.mean(data_nopref_5_overall1)))
# print('data_nopref_5_cost:{}'.format(np.mean(data_nopref_5_cost)))
# print('data_nopref_5_pref:{}'.format(np.mean(data_nopref_5_pref)))
# print('data_nopref_5_brisk:{}'.format(np.mean(data_nopref_5_brisk)))
# print('data_nopref_5_collision:{}'.format(np.mean(data_nopref_5_collision)))
# print('data_nopref_5_risk:{}'.format(np.mean(data_nopref_5_risk)))
# print('data_nopref_5_avoid:{}'.format(np.mean(data_nopref_5_avoid)))


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
# X values
# -----------------------------------------
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = np.arange(0, 6000, 1000)
xticks = ['TPAD', 'MINI', 'TMPUD', 'GLAD-SVM \n(ours)', 'GLAD \n(ours)', 'TPAD', 'MINI', 'TMPUD', 'GLAD-SVM \n(ours)', 'GLAD \n(ours)']

# -----------------------------------------
# plotting
# -----------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(params['figure_width'], params['figure_height']))
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')


# -----------------------------------------
# plot NoSafe
# -----------------------------------------
ax.bar(X[0] - params['bar_interval'], Y_nosafe_8, width=params['bar_width'], yerr=Y_nosafe_8_std, hatch=params['hatch'][0], color=params['color'][0], capsize=params['error_capsize'])
ax.bar(X[5] + params['bar_interval'], Y_nosafe_5, width=params['bar_width'], yerr=Y_nosafe_5_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])

# -----------------------------------------
# plot NoPref
# -----------------------------------------
ax.bar(X[2] - params['bar_interval'], Y_nopref_8, width=params['bar_width'], yerr=Y_nopref_8_std, hatch=params['hatch'][0], color=params['color'][0], capsize=params['error_capsize'])
ax.bar(X[7] + params['bar_interval'], Y_nopref_5, width=params['bar_width'], yerr=Y_nopref_5_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])


# -----------------------------------------
# plot NoCost
# -----------------------------------------
ax.bar(X[1] - params['bar_interval'], Y_nocost_8, width=params['bar_width'], yerr=Y_nocost_8_std, hatch=params['hatch'][0], color=params['color'][0], capsize=params['error_capsize'])
ax.bar(X[6] + params['bar_interval'], Y_nocost_5, width=params['bar_width'], yerr=Y_nocost_5_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])

# -----------------------------------------
# plot our (SVM)
# -----------------------------------------
ax.bar(X[3] - params['bar_interval'], Y_our_SVM_8, width=params['bar_width'], yerr=Y_our_SVM_8_std, hatch=params['hatch'][0], color=params['color'][0], capsize=params['error_capsize'])
ax.bar(X[8] + params['bar_interval'], Y_our_SVM_5, width=params['bar_width'], yerr=Y_our_SVM_5_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])

# -----------------------------------------
# plot our (ANN)
# -----------------------------------------
legend1 = ax.bar(X[4] - params['bar_interval'], Y_our_ANN_8, width=params['bar_width'], yerr=Y_our_ANN_8_std, hatch=params['hatch'][0], color=params['color'][0], capsize=params['error_capsize'])
legend2 = ax.bar(X[9] + params['bar_interval'], Y_our_ANN_5, width=params['bar_width'], yerr=Y_our_ANN_5_std, hatch=params['hatch'][1], color=params['color'][1], capsize=params['error_capsize'])

# -----------------------------------------
# xlabel
# -----------------------------------------
ax.set_xticks(X[:10])
ax.set_xlabel('Approach', fontsize=params['xlabel_fontsize'], fontweight=params['xlabel_fontweight'])
ax.set_xticklabels(xticks, fontsize=params['xaxis_fontsize'], fontweight=params['xaxis_fontweight'], rotation=params['xaxis_degree'])

# -----------------------------------------
# ylabel
# -----------------------------------------
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
ax.set_yticks(Y)
ax.set_ylabel('Utility', fontsize=params['ylabel_fontsize'], weight=params['ylabel_fontweight'])
yticks = ax.get_yticks()
yticks = yticks - params_eva['constant']
ax.set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')

# -----------------------------------------
# legend
# -----------------------------------------
ax.legend([legend1, legend2], [r'Heavy Traffic', r'Normal Traffic'], fontsize=params['legend_fontsize'])

# -----------------------------------------
# styles
# -----------------------------------------
fig.tight_layout()
ax.grid(params['grid'])

# -----------------------------------------
# show and save figure
# -----------------------------------------
# plt.show()
plt.savefig(params['fig_name'] + '.' + params['fig_svg'], format=params['fig_svg'], transparent=params['fig_transparent'])
cairosvg.svg2pdf(url=params['fig_name'] + '.' + params['fig_svg'], write_to=params['fig_name'] + '.' + params['fig_pdf'])
os.remove(params['fig_name'] + '.' + params['fig_svg'])