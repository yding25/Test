#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -----------------------------------------
# The entire code is to plot bar!
# -----------------------------------------

from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def sort_by_index(a, X):
    # sort bar 
    new_list = []
    for i in X:
       new_list.append(a[i])
    return new_list


# -----------------------------------------
# The collection is relevant to parameters
# -----------------------------------------
params = {
    'flag_sort': True, # True: bar is sorted
    }

# -----------------------------------------
# X values and their labels
# -----------------------------------------
X = [5, 6, 3, 9, 0, 8, 1, 7, 2, 4]
labels = ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']

# -----------------------------------------
# Y values for method 1
# -----------------------------------------
Y_method1 = [734.67,755,702.67,637.33,648,656.67,637.33,720.33,744.33,712]
Y_method1_std = [25.17,14,55.08,44.38,30.05,43.66,49.69,21.36,27.5,56.4]

# -----------------------------------------
# Y values for method 2
# -----------------------------------------
Y_method2 = [692.67,755,959,629.33,941.67,631.67,672,895.33,801.33,811]
Y_method2_std = [48.79,62.75,19,28.29,53.69,31.75,81.66, 17.01,50.12,88.1]

# -----------------------------------------
# Y values for method 3
# -----------------------------------------
Y_method3 = [845.67,864,952.67,783.67,965.67,614,710,902.33,849,795]
Y_method3_std = [25.48,21.7,58.77,84.4,3.06,71.25,89.21,10.07,25.98,31.43]


if params['flag_sort']:
    # method 1
    Y_method1 = sort_by_index(Y_method1, X)
    Y_method1_std = sort_by_index(Y_method1_std, X)
    
    # method 2
    Y_method2 = sort_by_index(Y_method2, X)
    Y_method2_std = sort_by_index(Y_method2_std, X)
    
    # method 3
    Y_method3 = sort_by_index(Y_method3, X)
    Y_method3_std = sort_by_index(Y_method3_std, X)

    # X labels
    labels = sort_by_index(labels, X)

# -----------------------------------------
# degrees of labels for X values
# -----------------------------------------
degrees = 45

# -----------------------------------------
# some settings
# -----------------------------------------
ind = np.arange(len(Y_method1))  # the x locations for the groups
width = 0.3  # the width of the bars
fig, ax = plt.subplots()
ax.bar(ind - width, Y_method1, width, yerr=Y_method1_std, label='Random', hatch = "//", color = '#AFBADC')
ax.bar(ind, Y_method2, width, yerr=Y_method2_std, label='Passive', hatch = "\\\\", color = '#F4E3B1')
ax.bar(ind + width, Y_method3, width, yerr=Y_method3_std, label='ITRS', hatch = "--", color = '#8ED3F4')

ax.set_ylabel('Accuracy (%)', fontsize= 14)
ax.set_xticks(ind)
ax.set_xticklabels(labels, fontsize = 14, rotation = degrees)
ax.legend()

fig.tight_layout()
plt.show()
