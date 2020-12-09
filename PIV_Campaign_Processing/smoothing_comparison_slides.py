# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:04:54 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import matplotlib.pyplot as plt
import post_processing_functions as ppf
from smoothn import smoothn as smo

Fol_In = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f750_1_p14_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_In)
Height, Width = ppf.get_img_shape(Fol_Raw)
Scale = Width/5
Dt = 1/ppf.get_frequency(Fol_Raw) # time between images
Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s

ppf.set_plot_parameters(20, 15, 10)
idx = 230
NX = ppf.get_column_amount(Fol_In)
x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, idx, NX)
u = u*Factor
v = v*Factor
x, y, u, v = ppf.pad(x, y, u, v, Width)
v_smo, dum, dum, dum = smo(v, s = 5)

y_ticks = np.arange(0,Height,4*Scale)
y_ticklabels = np.arange(0, 4*(Height/Width+1), 4, dtype = int)
x_ticks = np.linspace(0,Width, 6)
x_ticklabels = np.arange(0,6,1)

# v_smo[:,0] = 0
# v_smo[:,-1] = 0
# for i in range(0, 20):
fig, ax = plt.subplots(figsize = (8, 5))
ax.grid(b = True)
ax.plot(y[:,0],v[:,3], label = 'Unsmoothed')
ax.plot(y[:,0],v_smo[:,3], label = 'Smoothed')
ax.set_xticks(y_ticks)
ax.set_xticklabels(y_ticklabels)
ax.set_xlim(0,Height)
ax.set_ylim(-100,100)
ax.set_xlabel('$y$[mm]')
ax.set_ylabel('$v$[mm/s]')
ax.legend(prop={'size': 15}, loc = 'lower left')
fig.tight_layout(pad = 1.1)
fig.savefig('Vertical_profile.png', dpi = 400)
# ax.set_yticks()

fig, ax = plt.subplots(figsize = (8, 5))
ax.grid(b = True)
ax.plot(x[5,:],v[5,:], label = 'Unsmoothed')
ax.plot(x[5,:],v_smo[5,:], label = 'Smoothed')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticklabels)
ax.set_xlim(0,Width)
ax.set_ylim(-100,250)
ax.legend(prop={'size': 15}, loc = 'lower center')
ax.set_xlabel('$x$[mm]')
ax.set_ylabel('$v$[mm/s]')
fig.tight_layout(pad = 1.1)
fig.savefig('Horizontal_profile.png', dpi = 400)