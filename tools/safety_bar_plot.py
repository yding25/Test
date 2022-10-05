#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot bar!
# -----------------------------------------

from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import os

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X = [1, 2, 3, 4, 5]
labels = ['KNN', 'AdaBoost', 'LR', 'SVM', 'ANN']

# -----------------------------------------
# Y values for method 1, 2, 3, 4, 5
# -----------------------------------------
Y_method1 = [75.55]
Y_method1_std = [0]

Y_method2 = [79.74]
Y_method2_std = [0]

Y_method3 = [81.01]
Y_method3_std = [0.25]

Y_method4 = [83.64] # TBD
Y_method4_std = [0]

Y_method5 = [85.27]
Y_method5_std = [0.05]

# -----------------------------------------
# some settings
# -----------------------------------------
bar_width = 0.45
error_capsize = 5
xaxis_fontsize = 24
yaxis_fontsize = 24
legend_fontsize = 24
figure_weight = 8
figure_height = 5
xaxis_degrees = 0 # degrees of labels for X values
yaxis_degrees = 90
grid = True
ylim_min = 75
ylim_max = 86
fig_format1 = 'svg'
fig_format2 = 'pdf'
fig_transparent = False
filename = 'fig_5_safety_models'

# -----------------------------------------
# start plotting
# -----------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(figure_weight, figure_height))

ax.bar(X[0], Y_method1, width=bar_width, yerr=Y_method1_std, label=labels[0], hatch='o', color='#afbadc', ecolor='black', capsize=error_capsize)
ax.bar(X[1], Y_method2, width=bar_width, yerr=Y_method2_std, label=labels[1], hatch='xx', color='#f4e3b1', ecolor='black', capsize=error_capsize)
ax.bar(X[2], Y_method3, width=bar_width, yerr=Y_method3_std, label=labels[2], hatch='--', color='#8ed3f4', ecolor='black', capsize=error_capsize)
ax.bar(X[3], Y_method4, width=bar_width, yerr=Y_method4_std, label=labels[3], hatch='\\\\', color='#dabbdc', ecolor='black', capsize=error_capsize)
ax.bar(X[4], Y_method5, width=bar_width, yerr=Y_method5_std, label=labels[4], hatch='//', color='#cbe6b6', ecolor='black', capsize=error_capsize)

ax.set_xticks(X, fontsize=xaxis_fontsize)
ax.set_xticklabels(labels, fontsize=xaxis_fontsize, weight='bold', rotation=xaxis_degrees)
ax.set_ylabel('Accuracy (%)', fontsize=yaxis_fontsize, weight='bold', rotation=yaxis_degrees)
ax.legend(fontsize=legend_fontsize)

ax.grid(grid)
plt.xticks(fontsize=xaxis_fontsize)
plt.yticks(fontsize=yaxis_fontsize)
plt.ylim([ylim_min, ylim_max])
fig.tight_layout()

plt.show()
# plt.savefig(filename + '.' + fig_format1, format=fig_format1, transparent=fig_transparent)

# # save figure
# cairosvg.svg2pdf(url=filename + '.' + fig_format1, write_to=filename + '.' + fig_format2)
# os.remove(filename + '.' + fig_format1)