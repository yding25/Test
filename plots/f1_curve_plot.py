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
from matplotlib.pyplot import MultipleLocator

# -----------------------------------------
# styles
# -----------------------------------------
params = {
    # color
    # 'color': ['#C82423', '#2878B5', '#9AC9DB', '#F3D266', '#FF8884'],
      'color': ['#9AC9DB', '#F8AC8C', '#2878B5'],

    'ecolor': ['black'],
    
    'marker'     : ['d', '^', 'o', 's', 's', '>'], # refer to https: //matplotlib.org/stable/api/markers_api.html
    'marker_size': 22,
    
    'bar_width'   : 0.3,
    'bar_interval': 0.2,
    
    'error_capsize': 5,
    
    'xaxis_fontsize'   : 20,
    'xaxis_degree'     : 0,
    'xaxis_fontweight' : 'normal',
    'xlabel_fontsize'  : 22,
    'xlabel_fontweight': 'bold',
    'xlim_min'         : 0,
    'xlim_max'         : 10,

    'yaxis_fontsize'   : 20,
    'yaxis_degree'     : 90,
    'yaxis_fontweight' : 'normal',
    'ylabel_fontsize'  : 22,
    'ylabel_fontweight': 'bold',
    'ylim_min'         : 0,
    'ylim_max'         : 92,

    'legend_fontsize': 20,
    'legend_location': ['best', 'upper right'],
    
    'figure_width' : 10,
    'figure_height': 12,
    
    'grid': True,

    'fill_transparent': 0.5,

    'line_width': 3.,
    'line_style': ['-', '--', ':', '-.'],

    'size': np.array([300, 300, 300, 300, 300]),

    'hatch': ['\\\\', 'o', '//', '.', '-', 'x', 'o', 'O', '+', '*'],
    
    'fig_name'       : 'fig_safety',
    'fig_svg'        : 'svg',
    'fig_png'        : 'png',
    'fig_jpeg'       : 'jpeg',
    'fig_pdf'        : 'pdf',
    'fig_transparent': False,
}
params_eva = {
    'threshold_cost': 5000,  # if the task is completed within time threshold
    'penalty_cost'  : 0,     # cost penalty
    'penalty_risk'  : 15000, # risk penalty
    'constant'      : 8000,
}
params_debug = {
    'print': False
}

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X      = np.array([1, 2, 3, 4, 5, 6])
labels = ['16.7%', '33.3%', '50%', '66.7%', '83.3%', '100%']

# -----------------------------------------
# Y values for method 1, 2
# -----------------------------------------
# ANN, no_new_scenarios
Y_method1     = np.array([0.8004, 0.8267, 0.8307, 0.8381, 0.8438, 0.8448])
Y_method1_std = np.array([0, 0, 0, 0, 0, 0])
# ANN, new_scenarios
Y_method2     = np.array([0.7753, 0.7959, 0.8030, 0.8078, 0.8129, 0.8147])
Y_method2_std = np.array([0, 0, 0, 0, 0, 0])
# SVM, no_new_scenarios
Y_method3     = np.array([0.7734, 0.7931, 0.8100, 0.8166, 0.8246, 0.8265])
Y_method3_std = np.array([0, 0, 0, 0, 0, 0])
# SVM, new_scenarios
Y_method4     = np.array([0.7541, 0.7670, 0.7800, 0.7910, 0.7959, 0.7975])
Y_method4_std = np.array([0, 0, 0, 0, 0, 0])

# -----------------------------------------
# start plotting
# -----------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(params['figure_width'], params['figure_height']))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(X, Y_method1, marker=params['marker'][0], color=params['color'][0], label='SE-ANN', linewidth=params['line_width'], linestyle=params['line_style'][0], mec='black', markersize=params['marker_size'])
ax1.plot(X, Y_method3, marker=params['marker'][2], color=params['color'][1], label='SE-SVM', linewidth=params['line_width'], linestyle=params['line_style'][1], mec='black', markersize=params['marker_size'])

ax2 = plt.subplot(2, 1, 2)
ax2.plot(X, Y_method2, marker=params['marker'][1], color=params['color'][0], label='SE-ANN (#)', linewidth=params['line_width'], linestyle=params['line_style'][0], mec='black', markersize=params['marker_size'])
ax2.plot(X, Y_method4, marker=params['marker'][3], color=params['color'][1], label='SE-SVM (#)', linewidth=params['line_width'], linestyle=params['line_style'][1], mec='black', markersize=params['marker_size'])
# # ax.fill_between(X, Y_method1 - Y_method1_std, Y_method1 + Y_method1_std, alpha=params['fill_transparent'], color=params['color'][0])

ax1.set_xticks(X, fontsize=params['xaxis_fontsize'])
ax1.set_xticklabels(labels, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])
ax1.set_xlabel('Percentage of D$_{train}$', fontsize=params['xlabel_fontsize'], weight='bold')
ax1.set_ylabel(r'F1-score', fontsize=params['ylabel_fontsize'], weight='bold', rotation=params['yaxis_degree'])
ax1.legend(fontsize=params['legend_fontsize'])

ax1.yaxis.set_major_locator(MultipleLocator(0.015))
yticks = ax1.get_yticks()
print(yticks)
ax1.set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')



ax2.set_xticks(X, fontsize=params['xaxis_fontsize'])
ax2.set_xticklabels(labels, fontsize=params['xaxis_fontsize'], rotation=params['xaxis_degree'])
ax2.set_xlabel('Percentage of D$_{train}$', fontsize=params['xlabel_fontsize'], weight='bold')
ax2.set_ylabel(r'F1-score', fontsize=params['ylabel_fontsize'], weight='bold', rotation=params['yaxis_degree'])
ax2.legend(fontsize=params['legend_fontsize'])
# legend_properties = {'weight':'normal', 'size':legend_fontsize}
# plt.legend(prop=legend_properties)

ax2.yaxis.set_major_locator(MultipleLocator(0.015))
yticks = ax2.get_yticks()
print(yticks)
ax2.set_yticklabels(yticks, fontsize=params['yaxis_fontsize'], fontweight=params['yaxis_fontweight'], rotation=params['yaxis_degree'], verticalalignment='center')


ax1.grid(params['grid'])
ax2.grid(params['grid'])
plt.xticks(fontsize=params['xaxis_fontsize'])
plt.yticks(fontsize=params['yaxis_fontsize'])
# plt.ylim([params['ylim_min'], params['ylim_max']])
fig.tight_layout()
plt.subplots_adjust(wspace = 0, hspace = 0.25)

#plt.show()
plt.savefig(params['fig_name'] + '.' + params['fig_svg'], format=params['fig_svg'], transparent=params['fig_transparent'])
cairosvg.svg2pdf(url=params['fig_name'] + '.' + params['fig_svg'], write_to=params['fig_name'] + '.' + params['fig_pdf'])
os.remove(params['fig_name'] + '.' + params['fig_svg'])
