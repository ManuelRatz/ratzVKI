# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:46:08 2020

@author: manue
@description: Create animations for the slide. These are animations for
    the Fall F_h4_f1200_1_s
"""
import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
import os                               # for file paths
import numpy as np                      # for calculations
import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for some postprocessing stuff
import shutil                           # for deleting the folder with the gif images
import imageio                          # for animations
import warnings
# ignore warnings for the histograms, when there is a nan in them
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
Frame0 = 520# starting index of the run
Stp_T = 3 # step size in time
Seconds = 2 # how many seconds to observe the whole thing
# N_T = int((Seconds/Dt)/Stp_T)
N_T = 173
# N_T = 1

# these are the ticks and ticklabels to go from pixel -> mm for the coordinates
y_ticks = np.arange(0,Height,4*Scale)
y_ticklabels = np.arange(0, 4*(Height/Width+1), 4, dtype = int)
x_ticks = np.linspace(0,Width-1, 6)
x_ticklabels = np.arange(0,6,1)

# set up empty lists to append into and the names of the gifs
IMAGES_ROI = []
IMAGES_CONT = []
IMAGES_PROF = []
IMAGES_FLUX = []
IMAGES_HIGHPASS = []
IMAGES_HIST = []
Gif_Suffix = 'F0_%d_NT_%d_Stp_%d.gif' %(Frame0, N_T, Stp_T)
GIF_ROI = 'changing_roi' + Gif_Suffix
GIF_QUIV = 'contour_' + Gif_Suffix
GIF_PROF = 'profiles_' + Gif_Suffix
GIF_FLUX = 'flux_' + Gif_Suffix
GIF_HIGHPASS = 'highpass_filtered_' + Gif_Suffix
GIF_HIST = 'histogram_' + Gif_Suffix
# create a folder to store the images
Fol_Img = ppf.create_folder('images')

# load the data of the predicted height
h = ppf.load_h(Fol_In)
# set up a time array for plotting
t = np.linspace(0,Seconds,int(Seconds/Dt)+1)[::Stp_T]

# set a custom colormap
custom_map = ppf.custom_div_cmap(100, mincol='indigo', midcol='darkcyan' ,maxcol='yellow')

# enable or disable the plots
PLOT_ROI = False
PLOT_QUIV = False
PLOT_PROF = False
PLOT_FLUX = True
PLOT_HIGHPASS = False
PLOT_HIST = False


for i in range(0,N_T):
    print('Image %d of %d' %((i+1,N_T)))
    # calculate the loading index and load the data
    LOAD_IDX = Frame0 + Stp_T*i
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, LOAD_IDX, NX)

    x, y, u, v, ratio, valid, invalid = ppf.filter_invalid(x, y, u, v, ratio, mask, valid_thresh = 0.5)
    qfield = ppf.calc_qfield(x, y, u, v)    
    if PLOT_HIST == True:
        fig, ax = plt.subplots(figsize = (8, 5))
        ax.hist(ratio.ravel(), bins = 100, density = True)
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 0.3)
        ax.grid(b = True)
        ax.set_xlabel('Signal to noise ratio')
        ax.set_ylabel('Probability')
        ax.plot((5, 5), (0, 0.3), c = 'r', label = 'Threshold')
        ax.legend(loc = 'upper right')
        Name_Out = Fol_Img+os.sep+'s2n%06d.png'%LOAD_IDX  
        fig.savefig(Name_Out, dpi = 65)
        plt.close(fig)
        IMAGES_HIST.append(imageio.imread(Name_Out)) # append into list
    
    # convert to mm/s
    u = u*Factor
    v = v*Factor
    # pad the data using the no slip boundary condition
    x_pad, y_pad, u_pad, v_pad = ppf.pad(x, y, u, v, Width)   
    x_pco, y_pco = ppf.shift_grid(x, y)
    u_hp, v_hp = ppf.high_pass(u, v, sigma = 10, truncate = 3)
    img = ppf.load_raw_image(Fol_Raw, LOAD_IDX)
    
    # start with the animation of the changing ROI
    if PLOT_ROI == True:
        # create the figure
        fig, ax = plt.subplots(figsize = (4,8))
        ax.imshow(img, cmap=plt.cm.gray) # show the image
        ax.set_yticks(y_ticks) # set custom y ticks
        ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
        ax.set_xticks(x_ticks) # set custom x ticks 
        ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
        ax.set_xlabel('$x$[mm]') # set x label
        ax.set_ylabel('$y$[mm]') # set y label
        ax.set_ylim(0,Height-1)
        ax.set_xlim(0,Width-1)
        fig.tight_layout(pad=0.5) # crop edges of the figure to save space
        # plot a horizontal line of the predicted interface in case it is visible
        if h[LOAD_IDX] > 0:
            interface_line = Height-(np.ones((Width,1))*h[LOAD_IDX])
            ax.plot(interface_line, lw = 1, c='r')
        Name_Out = Fol_Img+os.sep+'roi%06d.png'%LOAD_IDX # set the output name
        fig.savefig(Name_Out, dpi = 65) # save the figure
        plt.close(fig) # close to not overcrowd
        IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list

    # plot the contour and the quiver, the principle is the same, so not everything is commented
    if PLOT_QUIV == True:
        fig, ax = plt.subplots(figsize = (4, 8))
        cs = plt.pcolormesh(x_pco,y_pco,v, vmin=-100, vmax=0, cmap = plt.cm.viridis) # create the contourplot using pcolormesh
        ax.set_aspect('equal') # set the correct aspect ratio
        clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
        clb.set_ticks(np.linspace(-100, 0, 6)) # set the colorbarticks
        clb.ax.set_title('Velocity \n [mm/s]', pad=15) # set the colorbar title
        STEPY= 4
        STEPX = 1
        plt.quiver(x[len(x)%2::STEPY, ::STEPX], y[len(x)%2::STEPY, ::STEPX], u[len(x)%2::STEPY, ::STEPX], v[len(x)%2::STEPY, ::STEPX],\
                    color='k', scale=1200, width=0.005,headwidth=4, headaxislength = 6)
        ax.set_yticks(np.arange(0,Height,4*Scale)) # set custom y ticks
        ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
        ax.set_xticks(x_ticks) # set custom x ticks 
        ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
        ax.set_xlabel('$x$[mm]') # set x label
        ax.set_ylabel('$y$[mm]') # set y label
        ax.set_ylim(0,Height-1)
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'contour%06d.png'%LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_CONT.append(imageio.imread(Name_Out))
        
    # plot the velocity profiles, the principle is the same, so not everything is commented
    if PLOT_PROF == True:
        fig, ax = plt.subplots(figsize=(8,5))
        # initialize array with values every 25% of the ROI
        Y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
        ax.set_title('$t$ = %03d [ms]' %(t[i]*1000))
        ax.plot(x_pad[Y_IND[0],:], v_pad[Y_IND[0],:], c='r', label='75\% ROI')
        ax.plot(x_pad[Y_IND[1],:], v_pad[Y_IND[1],:], c='b', label='50\% ROI')
        ax.plot(x_pad[Y_IND[2],:], v_pad[Y_IND[2],:], c='g', label='25\% ROI')
        # ax.scatter(x_pad[Y_IND[0],:], v_pad[Y_IND[0],:], c='r', marker='x', s=(300./fig.dpi)**2)
        # ax.scatter(x_pad[Y_IND[1],:], v_pad[Y_IND[1],:], c='b', marker='x', s=(300./fig.dpi)**2)
        # ax.scatter(x_pad[Y_IND[2],:], v_pad[Y_IND[2],:], c='g', marker='x', s=(300./fig.dpi)**2)
        ax.set_xlim(0, Width-1)
        ax.set_ylim(-120, 0)
        ax.set_xlabel('$x$[mm]')
        ax.set_ylabel('$v$[mm/s]')
        ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
        ax.grid(b=True)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'profiles%06d.png' %LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_PROF.append(imageio.imread(Name_Out))
        
    # plot the flux as a function of y, the principle is the same, so not everything is commented
    if PLOT_FLUX == True:
        fig, ax = plt.subplots(figsize=(9, 5))
        # integrate using trapz
        q = ppf.calc_flux(x_pad, v_pad, Scale)
        ax.set_title('$t$ = %03d [ms]' %(t[i]*1000))
        ax.scatter(y[:,0], q, c='r', marker='x', s=(300./fig.dpi)**2)
        ax.plot(y[:,0], q, c='r')
        ax.set_ylim(-500,0)
        ax.set_yticks(np.linspace(-500,0,5))
        ax.set_xlim(0, Height)
        ax.set_xticks(y_ticks)
        ax.set_xticklabels(y_ticklabels)
        ax.set_xlabel('$y$[mm]')
        ax.set_ylabel('$q$[mm$^2$/s]')
        ax.grid(b=True)
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'flux%06d.png'%LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_FLUX.append(imageio.imread(Name_Out))
    
    if PLOT_HIGHPASS == True:
        fig, ax = plt.subplots(figsize = (4, 8))
        ax.imshow(img, cmap=plt.cm.gray) # show the image
        # cs = plt.pcolormesh(x_pco, y_pco, qfield, cmap = plt.cm.viridis, vmin=-0.0003, vmax = 0.0005) # create the contourplot using pcolormesh
        # ax.set_aspect('equal') # set the correct aspect ratio
        # clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
        # # clb.set_ticks(np.linspace(-100, 0, 6)) # set the colorbarticks
        # clb.ax.set_title('Q Field \n [1/s$^2$]', pad=15) # set the colorbar title
        STEPY= 1
        STEPX = 1
        plt.quiver(x[(len(x)+1)%2::STEPY, ::STEPX], y[(len(x)+1)%2::STEPY, ::STEPX], u_hp[(len(x)+1)%2::STEPY, ::STEPX],\
                   v_hp[(len(x)+1)%2::STEPY, ::STEPX], color='lime', scale=350, width=0.005,headwidth=4, headaxislength = 6)
        ax.set_ylim(0,Height-1)
        ax.set_xlim(0,Width-1)
        ax.set_yticks(y_ticks) # set custom y ticks
        ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
        ax.set_xticks(x_ticks) # set custom x ticks 
        ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
        ax.set_xlabel('$x$[mm]')
        ax.set_ylabel('$y$[mm]')
        fig.tight_layout(pad=0.5)
        Name_Out = Fol_Img+os.sep+'highpass%06d.png'%LOAD_IDX
        fig.savefig(Name_Out, dpi=65)
        plt.close(fig)
        IMAGES_HIGHPASS.append(imageio.imread(Name_Out))
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
if PLOT_HIGHPASS == True:
    imageio.mimsave(GIF_HIGHPASS, IMAGES_HIGHPASS, duration=duration)
if PLOT_HIST == True:
    imageio.mimsave(GIF_HIST, IMAGES_HIST, duration=duration)
# delete the folder with the gif images
shutil.rmtree(Fol_Img)
#










































