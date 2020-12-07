# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:46:08 2020

@author: manue
@description: Create animations for the slide. These are animations for
    the Fall R_h2_f1200_1_p13 
"""

import os                               # for file paths
import numpy as np                      # for calculations
import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for some postprocessing stuff
import shutil                           # for deleting the folder with the gif images
import imageio                          # for animations
import warnings
warnings.filterwarnings("ignore")
# set the plot parameters
ppf.set_plot_parameters(20, 15, 20)

# give the input folder for the data
Fol_In = 'C:\PIV_Processed\Images_Processed\Fall_24_24_peak2RMS\Results_F_h4_f1200_1_s_24_24'
# give the input folder of the raw images (this is required to get the image width and frequency)
Fol_Raw = ppf.get_raw_folder(Fol_In)

# set the constants
Height, Width = ppf.get_img_shape(Fol_Raw) # image width in pixels
Scale = Width/5 # scaling factor in px/mm
Dt = 1/ppf.get_frequency(Fol_Raw) # time between images
Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s
NX = ppf.get_column_amount(Fol_In) # get the number of columns
# set frame0, the image step size and how many images to process
Frame0 = 354 # starting index of the run
Stp_T = 2 # step size in time
Seconds = 2 # how many seconds to observe the whole thing
# N_T = int((Seconds/Dt)/Stp_T)
N_T = 250

# set up empty lists to append into and the names of the gifs
IMAGES_ROI = []
IMAGES_H = []
IMAGES_CONT = []
IMAGES_PROF = []
IMAGES_FLUX = []
IMAGES_HIGHPASS = []
IMAGES_HIST = []
GIF_ROI = 'changing_roi_%d.gif' %(N_T)
GIF_H = 'h_%d.gif'%(N_T)
GIF_QUIV = 'contour_%d.gif'%(N_T)
GIF_PROF = 'profiles_%d.gif'%(N_T)
GIF_FLUX = 'flux_%d.gif'%(N_T)
GIF_HIGHPASS = 'highpass_filtered_%d.gif'%(N_T)
GIF_HIST = 'histogram_%d.gif'%(N_T)
# create a folder to store the images
Fol_Img = ppf.create_folder('images')

# load the data of the predicted height
h = ppf.load_h(Fol_In)
# set up a time array for plotting
t = np.linspace(0,Seconds,int(Seconds/Dt)+1)[::Stp_T]

# set a custom colormap
custom_map = ppf.custom_div_cmap(100, mincol='indigo', midcol='darkcyan' ,maxcol='yellow')

# enable or disable the plots
PLOT_ROI = True
PLOT_HT = True
PLOT_QUIV = True
PLOT_PROF = True
PLOT_FLUX = True
PLOT_HIGHPASS = True
PLOT_HIST = True

# set the initial displacement to zero for the plotting of the shifting ROI
disp = 0

for i in range(0,N_T):
    print('Image %d of %d' %((i+1,N_T)))
    # calculate the loading index and load the data
    LOAD_IDX = Frame0 + Stp_T*i
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, LOAD_IDX, NX)
    x, y, u, v, ratio, valid, invalid = ppf.filter_invalid(x, y, u, v, ratio, mask, valid_thresh = 0.9)
    # filter out the invalid rows 
    mask_nan = np.isfinite(ratio)
    mask = ~mask.astype('bool')
    valid = mask*mask_nan
    N_Rows = u.shape[0]
    for i in range(0,u.shape[0]):
        index = u.shape[0]-i-1
        if np.sum(valid[index,:])/valid.shape[0] < 0.5:
            v = v[index+2:]
            u = u[index+2:]
            x = x[index+2:]
            y = y[index+2:]
            ratio = ratio[index+2:]
            valid = valid[index+2:]
            break
    invalid = ~valid.astype('bool')

    # convert to mm/s
    u = u*Factor
    v = v*Factor
    # pad the data using the no slip boundary condition
    x_pad, y_pad, u_pad, v_pad = ppf.pad(x, y, u, v, Width)   
    x_pco, y_pco = ppf.shift_grid(x, y)
    u_hp, v_hp = ppf.high_pass(u, v, sigma = 3, truncate = 3)
    img = ppf.load_raw_image(Fol_Raw, LOAD_IDX)
    
    # start with the animation of the changing ROI
    if PLOT_ROI == True:
        name = Fol_Raw + os.sep + 'F_h1_f1200_1_q.%06d.tif' %(LOAD_IDX+1) # because we want frame_b
        img = imageio.imread(name)
        # create the figure
        fig, ax = plt.subplots(figsize = (2.5,8))
        ax.imshow(img, cmap=plt.cm.gray) # show the image
        ax.set_yticks(np.arange(img.shape[0]-20*Scale-1,img.shape[0],4*Scale)) # set custom y ticks
        ax.set_yticklabels(np.linspace(0,20,6,dtype=int)[::-1], fontsize=15) # set custom y ticklabels
        ax.set_xticks(np.linspace(0,Width-1, 6)) # set custom x ticks 
        ax.set_xticklabels(np.arange(0,6,1), fontsize=15) # set custom x ticklabels
        ax.set_xlabel('$x$[mm]', fontsize=20) # set x label
        ax.set_ylabel('$y$[mm]', fontsize=20) # set y label
        fig.tight_layout(pad=1.1) # crop edges of the figure to save space
        # plot a horizontal line of the predicted interface in case it is visible
        interface_line = np.ones((img.shape[1],1))*h[LOAD_IDX]
        ax.plot(interface_line, lw = 1, c='r')
        Name_Out = Fol_Img+os.sep+'roi%06d.png'%LOAD_IDX # set the output name
        fig.savefig(Name_Out, dpi = 65) # save the figure
        plt.close(fig) # close to not overcrowd
        IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list

    # plot the contour and the quiver, the principle is the same, so not everything is commented
    if PLOT_QUIV == True:
        fig, ax = plt.subplots(figsize = (4, 8))
        cs = plt.pcolormesh(x,y,v, vmin=-400, vmax=0, cmap = custom_map) # create the contourplot using pcolormesh
        ax.set_aspect('equal') # set the correct aspect ratio
        clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
        clb.set_ticks(np.arange(-400, 1, 40)) # set the colorbarticks
        clb.ax.set_title('Velocity \n [mm/s]', pad=15) # set the colorbar title
        STEPY= 4
        STEPX = 1
        plt.quiver(x[len(x)%2::STEPY, ::STEPX], y[len(x)%2::STEPY, ::STEPX], u[len(x)%2::STEPY, ::STEPX], v[len(x)%2::STEPY, ::STEPX],\
                    color='k', scale=1200, width=0.005,headwidth=4, headaxislength = 6)
        ax.set_yticks(np.arange(0,Height,4*Scale)) # set custom y ticks
        ax.set_yticklabels(np.linspace(0,20,6,dtype=int), fontsize=15) # set custom y ticklabels
        ax.set_xticks(np.linspace(0,Width-1, 6)) # set custom x ticks 
        ax.set_xticklabels(np.arange(0,6,1), fontsize=15) # set custom x ticklabels
        ax.set_xlabel('$x$[mm]', fontsize=20) # set x label
        ax.set_ylabel('$y$[mm]', fontsize=20) # set y label
        ax.set_ylim(0,img.shape[0])
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'contour%06d.png'%LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_CONT.append(imageio.imread(Name_Out))
        
    # plot the velocity profiles, the principle is the same, so not everything is commented
    if PLOT_PROF == True:
        fig, ax = plt.subplots(figsize=(8,5))
        # initialize array with values every 25% of the ROI
        y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
        ax.set_title('$t$ = %03d [ms]' %(t[i]*1000))
        ax.plot(x[y_IND[0],:], v[y_IND[0],:], c='r', label='75\% ROI')
        ax.plot(x[y_IND[1],:], v[y_IND[1],:], c='b', label='50\% ROI')
        ax.plot(x[y_IND[2],:], v[y_IND[2],:], c='g', label='25\% ROI')
        ax.scatter(x[y_IND[0],:], v[y_IND[0],:], c='r', marker='x', s=(300./fig.dpi)**2)
        ax.scatter(x[y_IND[1],:], v[y_IND[1],:], c='b', marker='x', s=(300./fig.dpi)**2)
        ax.scatter(x[y_IND[2],:], v[y_IND[2],:], c='g', marker='x', s=(300./fig.dpi)**2)
        ax.set_xlim(0, Width)
        ax.set_ylim(-400, 0)
        ax.set_xlabel('$x$[mm]', fontsize = 20)
        ax.set_ylabel('$v$[mm/s]', fontsize = 20)
        ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
        ax.grid(b=True)
        ax.set_xticks(np.linspace(0,Width-1, 6))
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
        ax.set_title('$t$ = %03d [ms]' %(t[i]*1000))
        ax.scatter(y[:,0], q, c='r', marker='x', s=(300./fig.dpi)**2)
        ax.plot(y[:,0], q, c='r')
        ax.set_ylim(-75000,0)
        ax.set_yticks(np.arange(-75000,1,12500))
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
duration = 0.05
if PLOT_FLUX == True:
    imageio.mimsave(GIF_FLUX, IMAGES_FLUX, duration=duration)
if PLOT_PROF == True:
    imageio.mimsave(GIF_PROF, IMAGES_PROF, duration=duration)
if PLOT_QUIV == True:
    imageio.mimsave(GIF_QUIV, IMAGES_CONT, duration=duration)
if PLOT_ROI == True:
    imageio.mimsave(GIF_ROI, IMAGES_ROI, duration=duration)
if PLOT_HT == True:
    imageio.mimsave(GIF_H, IMAGES_H, duration=duration)
if PLOT_HIGHPASS == True:
    imageio.mimsave(GIF_HIGHPASS, IMAGES_HIGHPASS, duration=duration)
if PLOT_HIST == True:
    imageio.mimsave(GIF_HIST, IMAGES_HIST, duration=duration)
# delete the folder with the gif images
shutil.rmtree(Fol_Img)
#










































