#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot curve!
# -----------------------------------------

from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
labels = ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']

# -----------------------------------------
# Y values for method 1
# -----------------------------------------
Y_method1 = np.array([734.67,755,702.67,637.33,648,656.67,637.33,720.33,744.33,712])
Y_method1_std = np.array([25.17,14,55.08,44.38,30.05,43.66,49.69,21.36,27.5,56.4])

# -----------------------------------------
# degrees of labels for X values
# -----------------------------------------
degrees = 45

# -----------------------------------------
# some settings
# -----------------------------------------
markers = ['v', '*', 'o']
colors = ['#ff8243', '#c043ff', '#82ff43']

ind = np.arange(len(Y_method1))  # the x locations for the groups
fig, ax = plt.subplots()
ax.plot(X, Y_method1, marker=markers[0], color=colors[0], label='Label', linewidth=2.0, mec='black', markersize=10)
ax.fill_between(X, Y_method1 - Y_method1_std, Y_method1 + Y_method1_std, alpha=0.5, color=colors[0])

ax.set_ylabel('Accuracy (%)', fontsize= 14)
ax.set_xticks(ind)
ax.set_xticklabels(labels, fontsize = 14, rotation = degrees)
ax.legend()

fig.tight_layout()
plt.grid(True)
plt.show()
