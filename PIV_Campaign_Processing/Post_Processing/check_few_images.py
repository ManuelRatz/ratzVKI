

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:46:08 2020

@author: manue
@description: Create animations for the slide. These are animations for
    the R_h2_f1200_1_p13 
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import os                               # for file paths
import numpy as np                      # for calculations
import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for some postprocessing stuff
import shutil                           # for deleting the folder with the gif images
import imageio                          # for animations

# set the plot parameters
ppf.set_plot_parameters()
settings = '_64_16'
Data_Location = 'C:\PIV_Processed' + os.sep
run_list = os.listdir(Data_Location + 'Images_Processed' + os.sep + 'Rise' + settings)
PLOT_ROI = True
PLOT_PROF = True



Fol_ROI_Gifs = ppf.create_folder(Data_Location + 'Images_Postprocessed' +settings + os.sep + 'ROI_Gifs' + os.sep)
Fol_Profile_Gifs = ppf.create_folder(Data_Location + 'Images_Postprocessed' +settings + os.sep + 'Profile_Gifs' + os.sep)
Fol_Images = ppf.create_folder(Data_Location + 'Images_Postprocessed' +settings + os.sep + 'Images' + os.sep)
Fol_Height = ppf.create_folder(Data_Location + 'Images_Postprocessed' +settings + os.sep + 'Height_Predictions' + os.sep)
for i in range(0,len(run_list)):
# for i in range(0,1):
    run = ppf.cut_processed_name(run_list[i])
    print('Exporting Run ' + run)
    # give the input folder for the data
    Fol_In = Data_Location + 'Images_Processed' + os.sep + 'Rise'+settings+ os.sep + 'Results_' + run +settings
    # give the input folder of the raw images (this is required to get the image width and frequency)
    Fol_Raw = Data_Location + 'Images_Preprocessed' + os.sep + run
    # create a folder to store the images
    GIF_ROI = Fol_ROI_Gifs + 'Changing_Roi_' + run + '.gif'
    GIF_PROF = Fol_Profile_Gifs + 'Profiles_' + run + '.gif'
    Fol_Gif_Images = ppf.create_folder(Fol_Images + os.sep + 'Images_'+run + os.sep)
    
    # set the constants
    IMG_HEIGHT, IMG_WIDTH= ppf.get_img_shape(Fol_Raw) # image width in pixels
    SCALE = IMG_WIDTH/5 # scaling factor in px/mm
    DT = 1/ppf.get_frequency(Fol_Raw) # time between images
    FACTOR = 1/(SCALE*DT) # conversion factor to go from px/frame to mm/s
    
    # load the data of the predicted height
    h = ppf.load_h(Fol_In)
    idx1 = np.nanargmax(h>0)
    idx2 = len(h)-np.nanargmax(h[::-1]!=0)
    h = h[idx1:idx2]
    h = (-h+IMG_HEIGHT)/SCALE
    # set up a time array for plotting
    t = np.linspace(0,len(h)*DT,len(h))
    
    # plot h as a function of time
    fig, ax = plt.subplots(figsize=(8,5))
    # crop the height before frame0
    # h_dum = h[FRAME0-1:]
    # plot the height, convert it to mm and shift to have the beginning height at the top of the image
    plot1 = ax.plot(t, h, c='r', label = 'Interface\nHeight')
    ax.plot(np.ones(len(h))*IMG_HEIGHT/SCALE, c = 'b')
    ax.set_title(ppf.separate_name(run))
    ax.set_ylim(0,np.nanmax(h)*1.05)
    ax.set_xlim(0,np.nanmax(t))
    ax.set_xlabel('$t$[s]', fontsize = 20)
    ax.set_ylabel('$h$[mm]', fontsize = 20)
    ax.grid(b=True)
    # ax.set_xticks(np.arange(0, 2.1 ,0.2))
    # ax.set_yticks(np.arange(0, 35, 5))
    fig.tight_layout(pad=1.1)
    ax.legend(loc='lower right')
    Name_Out = Fol_Height+os.sep+run+'.png'
    fig.savefig(Name_Out, dpi=150)
    plt.close(fig)
    
    NX = ppf.get_column_amount(Fol_In) # get the number of columns
    FRAME0 = idx1+1 # starting index of the run
    STP_SZ = 15 # step size in time
    SECONDS = (idx2-idx1)*DT # how many seconds to observe the whole thing
    N_T = int((SECONDS/DT)/STP_SZ)
    IMAGES_ROI = []
    IMAGES_PROF = []
    # N_T = 2
    # calculate the loading index and load the data
    for j in range(0, N_T):
        # print('Image %d of %d' %((j+1,N_T)))
        LOAD_IDX = FRAME0 + STP_SZ*j
        x, y, u, v, sig2noise, valid = ppf.load_txt(Fol_In, LOAD_IDX, NX)
        # convert to mm/s
        u = u*FACTOR
        v = v*FACTOR
        # start with the animation of the changing ROI
        if PLOT_ROI == True:
            name = Fol_Raw + os.sep + run+'.%06d.tif' %(LOAD_IDX+1) # because we want frame_b
            img = imageio.imread(name)
            # create the figure
            fig, ax = plt.subplots(figsize = (2.5,8))
            ax.set_title(ppf.separate_name(run))
            ax.imshow(img, cmap=plt.cm.gray) # show the image
            ax.set_yticks(np.arange(img.shape[0]-20*SCALE-1,img.shape[0],4*SCALE)) # set custom y ticks
            ax.set_yticklabels(np.linspace(0,20,6,dtype=int)[::-1], fontsize=15) # set custom y ticklabels
            ax.set_xticks(np.linspace(0,IMG_WIDTH-1, 6)) # set custom x ticks 
            ax.set_xticklabels(np.arange(0,6,1), fontsize=15) # set custom x ticklabels
            ax.set_xlabel('$x$[mm]', fontsize=20) # set x label
            ax.set_ylabel('$y$[mm]', fontsize=20) # set y label
            fig.tight_layout(pad=1.1) # crop edges of the figure to save space
            # plot a horizontal line of the predicted interface in case it is visible
            if ((h[LOAD_IDX-FRAME0]*SCALE)-IMG_HEIGHT) < 0:
                interface_line = np.ones((img.shape[1],1))*-((h[LOAD_IDX-FRAME0]*SCALE)-IMG_HEIGHT)
                ax.plot(interface_line, lw = 1, c='r')
            Name_Out = Fol_Gif_Images+os.sep+'roi_'+run +'_%06d.png'%LOAD_IDX # set the output name
            fig.savefig(Name_Out, dpi = 65) # save the figure
            plt.close(fig) # close to not overcrowd
            IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
            
        pad_0 = np.zeros((x.shape[0],1))
        pad_x_max = np.ones((x.shape[0],1))*IMG_WIDTH
        x = np.hstack((pad_0, x, pad_x_max))
        v = np.hstack((pad_0, v, pad_0))
        if j == 0:
            v_max = np.nanmax(v)
        if PLOT_PROF == True:
            fig, ax = plt.subplots(figsize=(8,5))
            # initialize array with values every 25% of the ROI
            y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
            ax.set_title(ppf.separate_name(run))
            ax.plot(x[y_IND[0],:], v[y_IND[0],:], c='r', label='5th Lowest')
            ax.plot(x[y_IND[1],:], v[y_IND[1],:], c='b', label='3rd Lowest')
            ax.plot(x[y_IND[2],:], v[y_IND[2],:], c='g', label='Lowest')
            ax.scatter(x[y_IND[0],:], v[y_IND[0],:], c='r', marker='x', s=(300./fig.dpi)**2)
            ax.scatter(x[y_IND[1],:], v[y_IND[1],:], c='b', marker='x', s=(300./fig.dpi)**2)
            ax.scatter(x[y_IND[2],:], v[y_IND[2],:], c='g', marker='x', s=(300./fig.dpi)**2)
            ax.set_xlim(0, IMG_WIDTH)
            ax.set_ylim(-0.6*v_max, v_max*1.05)
            ax.set_xlabel('$x$[mm]', fontsize = 20)
            ax.set_ylabel('$v$[mm/s]', fontsize = 20)
            ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
            ax.grid(b=True)
            ax.set_xticks(np.linspace(0, IMG_WIDTH, 6))
            ax.set_xticklabels(np.linspace(0,5,6, dtype=np.int))
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Images+os.sep+'profiles_'+run +'_%06d.png' %LOAD_IDX
            fig.savefig(Name_Out, dpi=65)
            plt.close(fig)
            IMAGES_PROF.append(imageio.imread(Name_Out))
    if PLOT_ROI == True:
        imageio.mimsave(GIF_ROI, IMAGES_ROI, duration = 0.075)
    if PLOT_PROF == True:
        imageio.mimsave(GIF_PROF, IMAGES_PROF, duration= 0.075)

# # plot the contour and the quiver, the principle is the same, so not everything is commented
# if PLOT_QUIV == True:
#     fig, ax = plt.subplots(figsize = (4, 8))
#     cs = plt.pcolormesh(x,y,v, vmin=-200, vmax=200, cmap = custom_map) # create the contourplot using pcolormesh
#     ax.set_aspect('equal') # set the correct aspect ratio
#     clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
#     clb.set_ticks(np.arange(-200, 201, 40)) # set the colorbarticks
#     clb.ax.set_title('Velocity \n [mm/s]', pad=15) # set the colorbar title
#     STEPY= 2
#     STEPX = 1
#     plt.quiver(x[len(x)%2::STEPY, ::STEPX], y[len(x)%2::STEPY, ::STEPX], u[len(x)%2::STEPY, ::STEPX], v[len(x)%2::STEPY, ::STEPX],\
#                color='k', scale=600, width=0.005,headwidth=4, headaxislength = 6)
#     ax.set_ylim(0,1271)
#     ax.set_yticks(np.arange(0,1300,220))
#     ax.set_yticklabels((0,4,8,12,16,20), fontsize=15)
#     ax.set_xlim(0,5)
#     ax.set_xticks(np.linspace(0,275,6))
#     ax.set_xticklabels(np.arange(0,6,1), fontsize=15)
#     ax.set_xlabel('$x$[mm]', fontsize=20)
#     ax.set_ylabel('$y$[mm]', fontsize=20)
#     fig.tight_layout(pad=0.5)
#     Name_Out = Fol_Img+os.sep+'contour%06d.png'%LOAD_IDX
#     fig.savefig(Name_Out, dpi=65)
#     plt.close(fig)

# # pad the data using the no slip boundary condition
# pad_0 = np.zeros((x.shape[0],1))
# pad_x_max = np.ones((x.shape[0],1))*IMG_WIDTH
# x = np.hstack((pad_0, x, pad_x_max))
# v = np.hstack((pad_0, v, pad_0))

# # plot the velocity profiles, the principle is the same, so not everything is commented
# if PLOT_PROF == True:
#     fig, ax = plt.subplots(figsize=(8,5))
#     # initialize array with values every 25% of the ROI
#     y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
#     ax.set_title('$t$ = %03d [ms]' %(t[i]*1000))
#     ax.plot(x[y_IND[0],:], v[y_IND[0],:], c='r', label='75\% ROI')
#     ax.plot(x[y_IND[1],:], v[y_IND[1],:], c='b', label='50\% ROI')
#     ax.plot(x[y_IND[2],:], v[y_IND[2],:], c='g', label='25\% ROI')
#     ax.scatter(x[y_IND[0],:], v[y_IND[0],:], c='r', marker='x', s=(300./fig.dpi)**2)
#     ax.scatter(x[y_IND[1],:], v[y_IND[1],:], c='b', marker='x', s=(300./fig.dpi)**2)
#     ax.scatter(x[y_IND[2],:], v[y_IND[2],:], c='g', marker='x', s=(300./fig.dpi)**2)
#     ax.set_xlim(0, IMG_WIDTH)
#     ax.set_ylim(-200, 200)
#     ax.set_xlabel('$x$[mm]', fontsize = 20)
#     ax.set_ylabel('$v$[mm/s]', fontsize = 20)
#     ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
#     ax.grid(b=True)
#     ax.set_xticks(np.linspace(0, 275, 6))
#     ax.set_xticklabels(np.linspace(0,5,6, dtype=np.int))
#     fig.tight_layout(pad=0.5)
#     Name_Out = Fol_Img+os.sep+'profiles%06d.png' %LOAD_IDX
#     fig.savefig(Name_Out, dpi=65)
#     plt.close(fig)
    
# # plot the flux as a function of y, the principle is the same, so not everything is commented
# if PLOT_FLUX == True:
#     fig, ax = plt.subplots(figsize=(9, 5))
#     # integrate using trapz
#     q = np.trapz(v, x)
#     ax.set_title('$t$ = %03d [ms]' %(t[i]*1000))
#     ax.scatter(y[:,0], q, c='r', marker='x', s=(300./fig.dpi)**2)
#     ax.plot(y[:,0], q, c='r')
#     ax.set_ylim(-30000,40000)
#     ax.set_yticks(np.arange(-30000,40001,10000))
#     ax.set_xlim(0, 1271)
#     ax.set_xticks(np.arange(0,1300,220))
#     ax.set_xticklabels(np.arange(0,21,4))
#     ax.set_xlabel('$y$[mm]', fontsize = 20)
#     ax.set_ylabel('$q$[mm$^2$/s]', fontsize = 20)
#     ax.grid(b=True)
#     fig.tight_layout(pad=0.5)
#     Name_Out = Fol_Img+os.sep+'flux%06d.png'%LOAD_IDX
#     fig.savefig(Name_Out, dpi=65)
#     plt.close(fig)




















































































