# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:46:08 2020

@author: manue
@description: Create animations for the slide. These are animations for
    the Rise R_h2_f1200_1_p13 
"""

import os                               # for file paths
import numpy as np                      # for calculations
import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for some postprocessing stuff
import shutil                           # for deleting the folder with the gif images
import imageio                          # for animations

# set the plot parameters
ppf.set_plot_parameters()

# give the input folder 
Fol_In = 'C:\PIV_Processed\Images_Processed\Results_R_h2_f1200_1_p13_Run_2'
Fol_Raw = 'C:\PIV_Processed\Images_Preprocessed\R_h2_f1200_1_p13'

IMG_WIDTH = 275
DT = 1/1200

# get the number of columns
NX = ppf.get_column_amount(Fol_In)

# set frame0 and how many images to process
FRAME0 = 391
STP_SZ = 4
# N_T = int((3000-FRAME0)/STP_SZ)
N_T = int(1200/STP_SZ)
# N_T = 2

# set up empty listss to append into and the names of the gifs
IMAGES_ROI = []
IMAGES_H = []
Gif_Roi = 'changing_roi.gif'
Gif_h = 'h.gif'

# create a folder to store the images
Fol_Img = ppf.create_folder('images')

# load the data of the predicted height
h = ppf.load_h(Fol_In)
# set up a time array for plotting
t = np.linspace(0,1,1201)[::STP_SZ]

for i in range(0,N_T):
    print('Image %d of %d' %((i+1,N_T)))
    LOAD_IDX = FRAME0 + STP_SZ*i
    
    # start with the animation of the changing ROI
    name = Fol_Raw + os.sep + 'R_h2_f1200_1_p13.%06d.tif' %(LOAD_IDX+1) # because we want frame_b
    img = imageio.imread(name)
    # create the figure
    fig, ax = plt.subplots(figsize = (2.5,8))
    ax.imshow(img, cmap=plt.cm.gray)
    # ax.axis('off')
    ax.set_yticks(np.arange(170,1300,220)) # set custom y ticks (not automatized)
    ax.set_yticklabels((20,16,12,8,4,0), fontsize=15) # set custom labels (not automatized)
    ax.set_xticks((0,55,110,165,220,274)) # set custom x ticks (not automatized)
    ax.set_xticklabels(np.arange(0,6,1), fontsize=15) # set custom y ticks (not automatized)
    ax.set_xlabel('$x$[mm]', fontsize=20)
    ax.set_ylabel('$y$[mm]', fontsize=20)
    fig.tight_layout(pad=1.1)
    if h[LOAD_IDX] > 0:
        interface_line = np.ones((img.shape[1],1))*h[LOAD_IDX]
        ax.plot(interface_line, lw = 1, c='r')
    Name_Out = Fol_Img+os.sep+'roi%06d.png'%LOAD_IDX
    fig.savefig(Name_Out, dpi = 65)
    plt.close(fig)
    IMAGES_ROI.append(imageio.imread(Name_Out))
    
    # now the plot for height over time
    fig, ax = plt.subplots(figsize=(8,5))
    h_dum = h[FRAME0-1:]
    h_dum = h_dum[::STP_SZ]
    plot1 = ax.plot(t[:i+1],(-h_dum[:i+1]+1271)/55, c='r', label = 'Interface\nHeight')
    ax.scatter(t[i],(-h_dum[i]+1271)/55, c='r', marker='x', s=(300./fig.dpi)**2)
    ax.set_ylim(0,30)
    ax.set_xlim(0,1)
    ax.set_xlabel('$t$[s]', fontsize = 20)
    ax.set_ylabel('$h$[mm]', fontsize = 20)
    ax.grid(b=True)
    ax.set_xticks(np.arange(0, 1.1 ,0.1))
    ax.set_yticks(np.arange(0, 35, 5))
    fig.tight_layout(pad=1.1)
    ax.legend(loc='upper right')
    Name_Out = Fol_Img+os.sep+'h%06d.png'%LOAD_IDX
    fig.savefig(Name_Out, dpi=65)
    plt.close(fig)
    IMAGES_H.append(imageio.imread(Name_Out))
    
# render the gifs
imageio.mimsave(Gif_Roi, IMAGES_ROI, duration = 0.05)
imageio.mimsave(Gif_h, IMAGES_H, duration = 0.05)
# delete the folder with the gif images
shutil.rmtree(Fol_Img)











































