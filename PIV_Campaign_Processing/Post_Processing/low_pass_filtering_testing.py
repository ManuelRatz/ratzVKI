# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:09:28 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import os
import post_processing_functions as ppf
import matplotlib.pyplot as plt

Fol_In = ''
NX = ppf.get_column_amount(Fol_In)

Index = 500
x, y, u, v, ratio, mask = ppf.loadtxt(Fol_In, Index, NX)

#%%
"""
Here goes the code for the Low pass filtering of the data to smooth the velocity field
"""



#%%
# figure to plot the profiles
fig, ax = plt.subplots()

# shift the grid for pcolormesh
x, y = ppf.shift_grid(x, y)

# figure for the contour plot
fig, ax = plt.subplots()
cs = plt.pcolormesh(x,y,v, vmin=-200, vmax=200, cmap = plt.cm.viridis) # create the contourplot using pcolormesh
ax.set_aspect('equal') # set the correct aspect ratio
clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
clb.set_ticks(np.arange(-1, 1, 40)) # set the colorbarticks
clb.ax.set_title('Velocity \n [px/frame]', pad=15) # set the colorbar title