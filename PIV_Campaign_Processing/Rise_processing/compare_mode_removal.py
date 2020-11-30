# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:51:02 2020

@author: manue
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import post_processing_functions as ppf

def load_txt(Fol_In, Mode):
    Fol_Load = Fol_In+ str(Mode) + '_modes_64_16' + os.sep + 'data_files' +os.sep+ 'field_000435.txt'
    Data = np.genfromtxt(Fol_Load)
    nx = 38
    N = Data.shape[0]
    ny = N //nx
    return Data[:, 0].reshape(nx, ny), Data[:, 1].reshape(nx, ny), Data[:, 4].reshape(nx, ny)
Fol_In = 'C:\PIV_Processed\Images_Processed\Results_R_h3_f1200_1_p13_'


# Fol_Load = Fol_In+ str(Mode) + '_modes_64_16' + os.sep + 'data_files' +os.sep+ 'field_000435.txt'
x, y, ratio1 = load_txt(Fol_In, 1)
x, y, ratio2 = load_txt(Fol_In, 2)
x, y, ratio3 = load_txt(Fol_In, 3)
x, y, ratio4 = load_txt(Fol_In, 4)
custom_map = ppf.custom_div_cmap(250, mincol='indigo', midcol='darkcyan' ,maxcol='yellow')



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (5, 5), sharex=True, sharey=True)
# set the title of each axis and create the pcolormesh plots
im1=ax1.pcolormesh(x, y, ratio1, vmin = 0, vmax = 10) # we name this for the colorbar later
ax2.pcolormesh(x, y, ratio2, vmin = 0, vmax = 10)
ax3.pcolormesh(x, y, ratio3, vmin = 0, vmax = 10)
ax4.pcolormesh(x, y, ratio4, vmin = 0, vmax = 10)

# disable the ticks and set the aspect ratio so we get square plots in the end
i = 0
for ax in fig.get_axes():
    ax.set_aspect(0.25)
    ax.axis('off')
    i = i+1
    ax.set_title('%d Modes removed'%i, fontsize = 10)
fig.suptitle('sig2noise ratio for different removed modes\n', fontsize=15, y=1.06)
fig.tight_layout(pad=1.1)
# create a list of the axes
axlist = [ax1,ax2,ax3,ax4]
# take the colorbar from the first plot (this is a global colorbar that will be displayed on the right side)
clb=fig.colorbar(im1, ax=axlist)  
# set the label for the colorbar
clb.ax.set_title('sig2noise\n ratio', fontsize=10)
# save the figure
Save_Name =  'sig2noise_ratio_mode_removal_comparison.png'
fig.savefig(Save_Name, dpi = 400)
# images_cont.append(imageio.imread(Save_Name))
plt.close(fig)