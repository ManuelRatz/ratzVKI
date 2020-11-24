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

# set the constants
IMG_WIDTH = 275 # image width in pixels
SCALE = IMG_WIDTH/5 # scaling factor in px/mm
DT = 1/1200 # time between two images
FACTOR = 1/(SCALE*DT) # conversion factor to go from px/frame to mm/s
# get the number of columns
NX = ppf.get_column_amount(Fol_In)

# set frame0, the image step size and how many images to process
FRAME0 = 391
STP_SZ = 4
# N_T = int((3000-FRAME0)/STP_SZ)
N_T = int(1200/STP_SZ)
# N_T = 1

# set up empty listss to append into and the names of the gifs
IMAGES_ROI = []
IMAGES_H = []
IMAGES_CONT = []
IMAGES_PROF = []
IMAGES_FLUX = []
GIF_ROI = 'changing_roi.gif'
GIF_H = 'h.gif'
GIF_QUIV = 'contour.gif'
GIF_PROF = 'profiles.gif'
GIF_FLUX = 'flux.gif'

# create a folder to store the images
Fol_Img = ppf.create_folder('images')

# load the data of the predicted height
h = ppf.load_h(Fol_In)
# set up a time array for plotting
t = np.linspace(0,1,1201)[::STP_SZ]

# set a custom colormap
custom_map = ppf.custom_div_cmap(100, mincol='indigo', midcol='darkcyan' ,maxcol='yellow')

# enable or disable the plots
PLOT_ROI = True
PLOT_HT = True
PLOT_QUIV = True
PLOT_PROF = True
PLOT_FLUX = True

for i in range(0,N_T):
    print('Image %d of %d' %((i+1,N_T)))
    # calculate the loading index and load the data
    LOAD_IDX = FRAME0 + STP_SZ*i
    x, y, u, v, sig2noise, valid = ppf.load_txt(Fol_In, LOAD_IDX, NX)
    # convert to mm/s
    u = u*FACTOR
    v = v*FACTOR
    # start with the animation of the changing ROI
    if PLOT_ROI == True:
        name = Fol_Raw + os.sep + 'R_h2_f1200_1_p13.%06d.tif' %(LOAD_IDX+1) # because we want frame_b
        img = imageio.imread(name)
        # create the figure
        fig, ax = plt.subplots(figsize = (2.5,8))
        ax.imshow(img, cmap=plt.cm.gray) # show the image
        ax.set_yticks(np.arange(170,1300,220)) # set custom y ticks (not automatized)
        ax.set_yticklabels((20,16,12,8,4,0), fontsize=15) # set custom labels (not automatized)
        ax.set_xticks((0,55,110,165,220,274)) # set custom x ticks (not automatized)
        ax.set_xticklabels(np.arange(0,6,1), fontsize=15) # set custom y ticks (not automatized)
        ax.set_xlabel('$x$[mm]', fontsize=20) # set x label
        ax.set_ylabel('$y$[mm]', fontsize=20) # set y label
        fig.tight_layout(pad=1.1) # crop edges of the figure to save space
        # plot a horizontal line of the predicted interface in case it is visible
        if h[LOAD_IDX] > 0:
            interface_line = np.ones((img.shape[1],1))*h[LOAD_IDX]
            ax.plot(interface_line, lw = 1, c='r')
        Name_Out = Fol_Img+os.sep+'roi%06d.png'%LOAD_IDX # set the output name
        fig.savefig(Name_Out, dpi = 65) # save the figure
        plt.close(fig) # close to not overcrowd
        IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
    # plot h as a function of time
    if PLOT_HT == True:
        # now the plot for height over time, the principle is the same, so not everything is commented
        fig, ax = plt.subplots(figsize=(8,5))
        # crop the height before frame0
        h_dum = h[FRAME0-1:]
        # shift it to only get the steps
        h_dum = h_dum[::STP_SZ]
        # plot the height, convert it to mm and shift to have the beginning height at the top of the image
        plot1 = ax.plot(t[:i+1],(-h_dum[:i+1]+1271)/SCALE, c='r', label = 'Interface\nHeight')
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
    
    # plot the contour and the quiver, the principle is the same, so not everything is commented
    if PLOT_QUIV == True:
        fig, ax = plt.subplots(figsize = (4, 8))
        cs = plt.pcolormesh(x,y,v, vmin=-200, vmax=200, cmap = custom_map) # create the contourplot using pcolormesh
        ax.set_aspect('equal') # set the correct aspect ratio
        clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
        clb.set_ticks(np.arange(-200, 201, 40)) # set the colorbarticks
        clb.ax.set_title('Velocity \n [mm/s]', pad=15) # set the colorbar title
        STEPY= 2
        STEPX = 1
        plt.quiver(x[len(x)%2::STEPY, ::STEPX], y[len(x)%2::STEPY, ::STEPX], u[len(x)%2::STEPY, ::STEPX], v[len(x)%2::STEPY, ::STEPX],\
                   color='k', scale=600, width=0.005,headwidth=4, headaxislength = 6)
        ax.set_ylim(0,1271)
        ax.set_yticks(np.arange(0,1300,220))
        ax.set_yticklabels((0,4,8,12,16,20), fontsize=15)
        ax.set_xlim(0,5)
        ax.set_xticks(np.linspace(0,275,6))
        ax.set_xticklabels(np.arange(0,6,1), fontsize=15)
        ax.set_xlabel('$x$[mm]', fontsize=20)
        ax.set_ylabel('$y$[mm]', fontsize=20)
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'contour%06d.png'%LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_CONT.append(imageio.imread(Name_Out))
    
    # pad the data using the no slip boundary condition
    pad_0 = np.zeros((x.shape[0],1))
    pad_x_max = np.ones((x.shape[0],1))*IMG_WIDTH
    x = np.hstack((pad_0, x, pad_x_max))
    v = np.hstack((pad_0, v, pad_0))
    
    # plot the velocity profiles, the principle is the same, so not everything is commented
    if PLOT_PROF == True:
        fig, ax = plt.subplots(figsize=(8,5))
        # initialize array with values every 25% of the ROI
        y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
        ax.plot(x[y_IND[0],:], v[y_IND[0],:], c='r', label='75\% ROI')
        ax.plot(x[y_IND[1],:], v[y_IND[1],:], c='b', label='50\% ROI')
        ax.plot(x[y_IND[2],:], v[y_IND[2],:], c='g', label='25\% ROI')
        ax.scatter(x[y_IND[0],:], v[y_IND[0],:], c='r', marker='x', s=(300./fig.dpi)**2)
        ax.scatter(x[y_IND[1],:], v[y_IND[1],:], c='b', marker='x', s=(300./fig.dpi)**2)
        ax.scatter(x[y_IND[2],:], v[y_IND[2],:], c='g', marker='x', s=(300./fig.dpi)**2)
        ax.set_xlim(0, IMG_WIDTH)
        ax.set_ylim(-200, 200)
        ax.set_xlabel('$x$[mm]', fontsize = 20)
        ax.set_ylabel('$v$[mm/s]', fontsize = 20)
        ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
        ax.grid(b=True)
        ax.set_xticks(np.linspace(0, 275, 6))
        ax.set_xticklabels(np.linspace(0,5,6, dtype=np.int))
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'profiles%06d.png' %LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_PROF.append(imageio.imread(Name_Out))
        
    # plot the flux as a function of y, the principle is the same, so not everything is commented
    if PLOT_FLUX == True:
        fig, ax = plt.subplots(figsize=(9, 5))
        # integrate using trapz
        q = np.trapz(v, x)
        ax.scatter(y[:,0], q, c='r', marker='x', s=(300./fig.dpi)**2)
        ax.plot(y[:,0], q, c='r')
        ax.set_ylim(-30000,40000)
        ax.set_yticks(np.arange(-30000,40001,10000))
        ax.set_xlim(0, 1271)
        ax.set_xticks(np.arange(0,1300,220))
        ax.set_xticklabels(np.arange(0,21,4))
        ax.set_xlabel('$y$[mm]', fontsize = 20)
        ax.set_ylabel('$q$[mm$^2$/s]', fontsize = 20)
        ax.grid(b=True)
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'flux%06d.png'%LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_FLUX.append(imageio.imread(Name_Out))
# render the gifs
imageio.mimsave(GIF_FLUX, IMAGES_FLUX, duration=0.05)
imageio.mimsave(GIF_PROF, IMAGES_PROF, duration=0.05)
imageio.mimsave(GIF_QUIV, IMAGES_CONT, duration=0.05)
imageio.mimsave(GIF_ROI, IMAGES_ROI, duration = 0.05)
imageio.mimsave(GIF_H, IMAGES_H, duration = 0.05)
# delete the folder with the gif images
shutil.rmtree(Fol_Img)











































