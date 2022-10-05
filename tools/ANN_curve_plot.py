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

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
X_label = np.arange(1, 45, 5)
labels = X_label

# -----------------------------------------
# Y values for method 1, 2
# -----------------------------------------
Y_method1 = np.array([72.76, 79.37, 80.65, 80.91, 81.92, 82.14, 82.58, 82.70, 83.23, 83.15, 83.28, 84.01, 84.09, 84.50, 85.13, 84.98, 85.26, 85.20, 85.75, 86.54, 85.95, 86.32, 86.84, 86.80, 86.71, 86.95, 86.99, 87.36, 87.36, 87.19, 88.82, 89.24, 89.44, 89.35, 89.55, 89.60, 90.28, 89.60, 90.02, 90.05])

Y_method2 = np.array([80.78, 81.04, 81.33, 81.91, 81.36, 82.05, 82.51, 83.12, 83.27, 82.80, 82.89, 83.47, 82.40, 83.76, 83.67, 83.99, 84.05, 84.34, 82.69, 84.25, 84.05, 84.31, 84.45, 84.68, 83.55, 83.32, 84.71, 84.42, 84.42, 84.34, 85.29, 85.12, 85.00, 85.23, 85.20, 85.09, 85.06, 85.12, 85.20, 85.23])

# -----------------------------------------
# some settings
# -----------------------------------------
markers = ['v', '*', 'o']
marker_size = 12
colors = ['#cbe6b6', '#ff8243', '#c043ff', '#82ff43']
bar_width = 0.45
error_capsize = 5
xaxis_fontsize = 12
xlabel_fontsize = 12
yaxis_fontsize = 12
ylabel_fontsize = 12
legend_fontsize = 12
figure_weight = 8
figure_height = 4
xaxis_degrees = 0 # degrees of labels for X values
yaxis_degrees = 90
grid = True
ylim_min = 70
ylim_max = 92
fig_format1 = 'svg'
fig_format2 = 'pdf'
fig_transparent = False
fill_transparent = 0.5
line_width = 2
filename = 'fig_ANN'

# -----------------------------------------
# start plotting
# -----------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(figure_weight, figure_height))

ax.plot(X, Y_method1, marker=markers[0], color=colors[0], label='Train', linewidth=line_width, mec='black', markersize=marker_size)
ax.fill_between(X, Y_method1, Y_method1, alpha=fill_transparent, color=colors[0])
ax.plot(X, Y_method2, marker=markers[1], color=colors[1], label='Test', linewidth=line_width, mec='black', markersize=marker_size)
ax.fill_between(X, Y_method2, Y_method2, alpha=fill_transparent, color=colors[1])

ax.set_xticks(X_label, fontsize=xaxis_fontsize)
ax.set_xticklabels(labels, fontsize=xaxis_fontsize, rotation=xaxis_degrees)
ax.set_xlabel('Epoch', fontsize=xlabel_fontsize, weight='bold', rotation=xaxis_degrees)
ax.set_ylabel('Accuracy (%)', fontsize=ylabel_fontsize, weight='bold', rotation=yaxis_degrees)
ax.legend(fontsize=legend_fontsize)
# legend_properties = {'weight':'normal', 'size':legend_fontsize}
# plt.legend(prop=legend_properties)

ax.grid(grid)
plt.xticks(fontsize=xaxis_fontsize)
plt.yticks(fontsize=yaxis_fontsize)
plt.ylim([ylim_min, ylim_max])
plt.text(20, 80, r'Epoch=35, Precision:0.849, Recall:0.837', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8})
ax.arrow(30, 81, 4, 3.5, width=0.2, head_width=0.6)
fig.tight_layout()

# plt.show()
plt.savefig(filename + '.' + fig_format1, format=fig_format1, transparent=fig_transparent)

# save figure
cairosvg.svg2pdf(url=filename + '.' + fig_format1, write_to=filename + '.' + fig_format2)
os.remove(filename + '.' + fig_format1)
