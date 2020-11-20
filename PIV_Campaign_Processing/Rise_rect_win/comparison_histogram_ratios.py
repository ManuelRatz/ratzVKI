# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:39:23 2020

@author: manue
"""
import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for reshaping the arrays
import os                               # for file paths
import numpy as np
import imageio

# set the plot parameters
ppf.set_plot_parameters()

# give the input folders
Fol_In_1 = 'C:\PIV_Processed\Images_Processed\Results_Run_1_R_h1_f1200_1_p15'
Fol_In_2 = 'C:\PIV_Processed\Images_Processed\Results_Run_2_R_h1_f1200_1_p15'
Fol_In_3 = 'C:\PIV_Processed\Images_Processed\Results_Run_3_R_h1_f1200_1_p15'
Fol_In_4 = 'C:\PIV_Processed\Images_Processed\Results_Run_4_R_h1_f1200_1_p15'
Fol_In_5 = 'C:\PIV_Processed\Images_Processed\Results_Run_5_R_h1_f1200_1_p15'
Fol_In_6 = 'C:\PIV_Processed\Images_Processed\Results_Run_6_R_h1_f1200_1_p15'

Fol_Hist = 'fol_hist'
if not os.path.exists(Fol_Hist):
    os.makedirs(Fol_Hist)
GIFNAME_HIST = 'Histograms.gif'
images_hist = []

Fol_Cont = 'fol_cont'
if not os.path.exists(Fol_Cont):
    os.makedirs(Fol_Cont)
GIFNAME_CONT = 'Contours.gif'
images_cont = []

# set the frame to look at
frame0 = 279
n_t = 20
# this is in case we want to loop over multiple images
for i in range(0, n_t):
    print('Image %d of %d' %((i+1, n_t)))
    # calculate the loading index
    load_index = frame0 + 10*i
    # load the number of columns
    nx1 = ppf.get_column_amount(Fol_In_1)
    nx2 = ppf.get_column_amount(Fol_In_2)
    nx3 = ppf.get_column_amount(Fol_In_3)
    nx4 = ppf.get_column_amount(Fol_In_4)
    nx5 = ppf.get_column_amount(Fol_In_5)
    nx6 = ppf.get_column_amount(Fol_In_6)
    
    
    # load the ratios separately (this is required because plt.hist does not like 2d arrays)
    ratio1 = np.genfromtxt(Fol_In_1 + os.sep + 'data_files' + os.sep + 'field_%06d.txt'% load_index)[:,4]
    ratio2 = np.genfromtxt(Fol_In_2 + os.sep + 'data_files' + os.sep + 'field_%06d.txt'% load_index)[:,4] 
    ratio3 = np.genfromtxt(Fol_In_3 + os.sep + 'data_files' + os.sep + 'field_%06d.txt'% load_index)[:,4]
    ratio4 = np.genfromtxt(Fol_In_4 + os.sep + 'data_files' + os.sep + 'field_%06d.txt'% load_index)[:,4]
    ratio5 = np.genfromtxt(Fol_In_5 + os.sep + 'data_files' + os.sep + 'field_%06d.txt'% load_index)[:,4]
    ratio6 = np.genfromtxt(Fol_In_6 + os.sep + 'data_files' + os.sep + 'field_%06d.txt'% load_index)[:,4]
    
    # create a figure with 4 plots as subfigure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (8, 5), sharex=True, sharey=True)
    # set the limits
    ax1.set_xlim(0,6)
    ax1.set_ylim(0,180)
    # set the titles of each axis and plot the histogram
    ax1.set_title('(256,128,96); (64,32,24)', fontsize = 10)
    ax1.hist(ratio1, range=(0,6), bins = 50)
    ax1.set_xticks(np.arange(0,7,1))
    ax2.set_title('(256,128,64); (64,32,16)', fontsize = 10)
    ax2.hist(ratio2, range=(0,6), bins=50)
    ax3.set_title('(128,64,48); (32,16,12)', fontsize = 10)
    ax3.hist(ratio3, range=(0,6), bins = 50)
    ax4.set_title('(256,128,64); (16,16,16)', fontsize = 10)
    ax4.hist(ratio4, range=(0,6), bins = 50)
    ax5.set_title('(128,64,32); (64,32,16)', fontsize = 10)
    ax5.hist(ratio5, range=(0,6), bins = 50)
    ax6.set_title('(256,128,64,32); (64,32,32,32)', fontsize = 10)
    ax6.hist(ratio6, range=(0,6), bins = 50)
    for ax in fig.get_axes():
        # enable the grid for each plot
        ax.grid(b=True,lw=1)
    # save the figure
    Save_Name = Fol_Hist + os.sep + 'histogram_%06d.jpg' %load_index
    fig.savefig(Save_Name, dpi=100)
    images_hist.append(imageio.imread(Save_Name))
    plt.close(fig)
    
    # load the x, y coordinates and the ratio
    x1,y1,u1,v1,ratio1,mask1 = ppf.load_txt(Fol_In_1, load_index, nx1)
    x2,y2,u2,v2,ratio2,mask2 = ppf.load_txt(Fol_In_2, load_index, nx2)
    x3,y3,u3,v3,ratio3,mask3 = ppf.load_txt(Fol_In_3, load_index, nx3)
    x4,y4,u4,v4,ratio4,mask4 = ppf.load_txt(Fol_In_4, load_index, nx4)
    x5,y5,u5,v5,ratio5,mask5 = ppf.load_txt(Fol_In_5, load_index, nx5)
    x6,y6,u6,v6,ratio6,mask6 = ppf.load_txt(Fol_In_6, load_index, nx6)
    
    # design a custom colormap for the pcolormesh plots
    custom_map = ppf.custom_div_cmap(250, mincol='indigo', midcol='darkcyan' ,maxcol='yellow')
    
    # create a figure with 4 plots as subfigure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (5, 5), sharex=True, sharey=True)
    # set the title of each axis and create the pcolormesh plots
    im1=ax1.pcolormesh(x1,y1,ratio1, vmin=0, vmax=10) # we name this for the colorbar later
    ax1.set_title('(256,128,96);\n(64,32,24)', fontsize = 10)
    ax2.pcolormesh(x2,y2,ratio2, vmin=0, vmax=10)
    ax2.set_title('(256,128,64);\n(64,32,16)', fontsize = 10)
    ax3.pcolormesh(x3,y3,ratio3, vmin=0, vmax=10)
    ax3.set_title('(128,64,48);\n(32,16,12)', fontsize = 10)
    ax4.pcolormesh(x4,y4,ratio4, vmin=0, vmax=10)
    ax4.set_title('(256,128,64);\n(16,16,16)', fontsize = 10)
    ax5.pcolormesh(x5,y5,ratio5, vmin=0, vmax=10)
    ax5.set_title('(128,64,32);\n(64,32,16)', fontsize = 10)
    ax6.pcolormesh(x6,y6,ratio6, vmin=0, vmax=10)
    ax6.set_title('(256,128,64,32);\n(64,32,32,32)', fontsize = 10)
    # disable the ticks and set the aspect ratio so we get square plots in the end
    for ax in fig.get_axes():
        ax.set_aspect(0.25)
        ax.axis('off')
    # create a list of the axes
    axlist = [ax1,ax2,ax3,ax4,ax5,ax6]
    # take the colorbar from the first plot (this is a global colorbar that will be displayed on the right side)
    clb=fig.colorbar(im1, ax=axlist)  
    # set the label for the colorbar
    clb.ax.set_title('sig2noise\n ratio', fontsize=10)
    # save the figure
    Save_Name = Fol_Cont + os.sep + 'contour_%06d.jpg' %load_index
    fig.savefig(Save_Name, dpi = 250)
    images_cont.append(imageio.imread(Save_Name))
    plt.close(fig)
    
imageio.mimsave(GIFNAME_HIST, images_hist, duration = 0.75) 
imageio.mimsave(GIFNAME_CONT, images_cont, duration = 0.75)

import shutil
shutil.rmtree(Fol_Hist)
shutil.rmtree(Fol_Cont)  
    
    
    
    
    
    
    
    
    
    
    
    
    