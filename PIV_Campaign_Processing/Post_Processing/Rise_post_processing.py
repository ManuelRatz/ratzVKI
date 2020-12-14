

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
ppf.set_plot_parameters(20, 15, 10)

# here a list of all the runs that will be processed is created
settings = '_64_16'
Data_Location = 'C:\PIV_Processed' + os.sep
run_list = os.listdir(Data_Location + 'Images_Processed' + os.sep + 'Rise' + settings + '_peak2RMS')
# create a postprocessing folder to store the results
Fol_PP = ppf.create_folder('C:\PIV_Processed\Images_Postprocessed' + settings + os.sep)

##############################################################################
######## enable or disable the creation of individual gifs ###################
##############################################################################

PLOT_ROI = False
PLOT_PROF = False
PLOT_H = False
PLOT_FLUX = False

##############################################################################
################ set some cosmetics ##########################################
##############################################################################

Stp_T = 50 # step size in time
duration = 0.05 # duration of each image in the gif


##############################################################################
################ iterate over all the runs ###################################
##############################################################################

# for i in range(0,len(run_list)):
for i in range(0,1):
    run = ppf.cut_processed_name(run_list[i])
    Fol_Out = ppf.create_folder(Fol_PP + run + os.sep)
    print('Exporting Run ' + run)
    # give the input folder for the data
    Fol_In = Data_Location + 'Images_Processed' + os.sep + 'Rise'+settings + '_peak2RMS'+ os.sep + 'Results_' + run +settings
    # give the input folder of the raw images (this is required to get the image width and frequency)
    Fol_Raw = ppf.get_raw_folder(Fol_In)
    
    # set the constants of the images to go from px/frame to mm/s
    Height, Width = ppf.get_img_shape(Fol_Raw) # image width in pixels
    Scale = Width/5 # scaling factor in px/mm
    Dt = 1/ppf.get_frequency(Fol_Raw) # time between images
    Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s
    
    # load the data of the predicted height
    h = ppf.load_h(Fol_In)
    idx1 = np.nanargmax(h>0)
    idx2 = len(h)-np.nanargmax(h[::-1]!=0)
    # use a dummy to calculate the indices of valid images (that were processed)
    h_time = h[idx1:idx2]
    # set up a time array for plotting
    t = np.linspace(0,len(h_time)*Dt,len(h_time))
    
    # calculate the number of columns and some other constants
    NX = ppf.get_column_amount(Fol_In) # get the number of columns
    Frame0 = idx1+1 # starting index of the run
    Seconds = (idx2-idx1)*Dt # how many seconds to observe the whole thing
    N_T = int((Seconds/Dt)/Stp_T)
    
    # these are the ticks and ticklabels to go from pixel -> mm for the coordinates
    y_ticks = np.arange(0,Height,4*Scale)
    y_ticklabels = np.arange(0, 4*(Height/Width+1), 4, dtype = int)
    x_ticks = np.linspace(0,Width-1, 6)
    x_ticklabels = np.arange(0,6,1)
    
    # create the gif names
    Gif_Suffix = '_' + run + '.gif'
    GIF_ROI = Fol_Out + 'Changing_Roi' + Gif_Suffix
    GIF_PROF = Fol_Out + 'Profiles' + Gif_Suffix
    GIF_H = Fol_Out + 'H_of_t' + Gif_Suffix
    # create a folder to store the images
    Fol_Gif_ROI = ppf.create_folder(Fol_Out + os.sep + 'Images_ROI_'+run + os.sep)
    Fol_Gif_Hist = ppf.create_folder(Fol_Out + os.sep + 'Images_Hist_'+run + os.sep)
    Fol_Gif_H = ppf.create_folder(Fol_Out + os.sep + 'Images_h_'+run + os.sep)
    Fol_Gif_Prof = ppf.create_folder(Fol_Out + os.sep + 'Images_Prof_'+run + os.sep)
    Fol_Gif_Prof = ppf.create_folder(Fol_Out + os.sep + 'Images_Flux_'+run + os.sep)
    
    # create empty lists to append the images into
    IMAGES_ROI = []
    IMAGES_PROF = []
    IMAGES_H = []
    IMAGES_FLUX = []
    # iterate over all the iamges in the current run
    N_T = 5
    # calculate the loading index and load the data
    for j in range(0, N_T):
        # print('Image %d of %d' %((j+1,N_T))) # update the user
        # calculate the loading index and load the data
        LOAD_IDX = Frame0 + Stp_T*j
        x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, LOAD_IDX, NX)
        # filter out the invalid rows
        # x, y, u, v, ratio, valid, invalid = ppf.filter_invalid(x, y, u, v, ratio, mask, valid_thresh = 0.5)    
        # convert to mm/s
        u = u*Factor
        v = v*Factor
        # pad the data using the no slip boundary condition
        x_pad, y_pad, u_pad, v_pad = ppf.pad(x, y, u, v, Width)   
        x_pco, y_pco = ppf.shift_grid(x, y)
        u_hp, v_hp = ppf.high_pass(u, v, sigma = 3, truncate = 3)
        img = ppf.load_raw_image(Fol_Raw, LOAD_IDX)
        
        # plots of the changing ROI
        if PLOT_ROI == True:
            # create the figure
            fig, ax = plt.subplots(figsize = (2.5,8))
            ax.set_ylim(0,Height)
            ax.set_xlim(0,Width)
            ax.imshow(img, cmap=plt.cm.gray) # show the image
            ax.set_yticks(y_ticks) # set custom y ticks
            ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
            ax.set_xticks(x_ticks) # set custom x ticks 
            ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
            ax.set_xlabel('$x$[mm]') # set x label
            ax.set_ylabel('$y$[mm]') # set y label
            fig.tight_layout(pad=0.5) # crop edges of the figure to save space
            # plot a horizontal line of the predicted interface in case it is visible
            if h[LOAD_IDX] > 0:
                interface_line = np.ones((img.shape[1],1))*(Height-h[LOAD_IDX])
                ax.plot(interface_line, lw = 1, c='r')
            Name_Out = Fol_Gif_ROI+os.sep+'roi%06d.png'%LOAD_IDX # set the output name
            fig.savefig(Name_Out, dpi = 65) # save the figure
            plt.close(fig) # close to not overcrowd
            IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
            
        # plots of the changing height
        if PLOT_H == True:
            # now the plot for height over time, the principle is the same, so not everything is commented
            fig, ax = plt.subplots(figsize=(8,5))
            # crop the height before frame0
            h_dum = h[Frame0-1:]
            # shift it to only get the steps
            h_dum = h_dum[::Stp_T]
            # plot the height, convert it to mm and shift to have the beginning height at the top of the image
            ax.plot(t[:j+1],(-h_dum[:j+1]+Height)/Scale, c='r', label = 'Interface\nHeight')
            ax.set_title('$t$ = %03d [ms]' %(t[j]*1000))
            ax.scatter(t[j],(-h_dum[j]+Height)/Scale, c='r', marker='x', s=(300./fig.dpi)**2)
            ax.set_ylim(0,30)
            ax.set_xlim(0,1)
            ax.set_xlabel('$t$[s]')
            ax.set_ylabel('$h$[mm]')
            ax.grid(b=True)
            ax.set_xticks(np.arange(0, 2.1 ,0.2))
            ax.set_yticks(np.arange(0, 35, 5))
            fig.tight_layout(pad=1.1)
            ax.legend(loc='upper right')
            Name_Out = Fol_Gif_H + os.sep+'h%06d.png'%LOAD_IDX
            fig.savefig(Name_Out, dpi=65)
            plt.close(fig)
            IMAGES_H.append(imageio.imread(Name_Out))
         
        # plots of the velocity profiles    
        if j == 0:
            v_max = np.nanmax(v)
        if PLOT_PROF == True:
            fig, ax = plt.subplots(figsize=(8,5))
            # initialize array with values every 25% of the ROI
            Y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
            ax.set_title(ppf.separate_name(run))
            ax.plot(x_pad[Y_IND[0],:], v_pad[Y_IND[0],:], c='r', label='5th Lowest')
            ax.plot(x_pad[Y_IND[1],:], v_pad[Y_IND[1],:], c='b', label='3rd Lowest')
            ax.plot(x_pad[Y_IND[2],:], v_pad[Y_IND[2],:], c='g', label='Lowest')
            # ax.scatter(x[y_IND[0],:], v[y_IND[0],:], c='r', marker='x', s=(300./fig.dpi)**2)
            # ax.scatter(x[y_IND[1],:], v[y_IND[1],:], c='b', marker='x', s=(300./fig.dpi)**2)
            # ax.scatter(x[y_IND[2],:], v[y_IND[2],:], c='g', marker='x', s=(300./fig.dpi)**2)
            ax.set_xlim(0, Width)
            ax.set_ylim(-0.6*v_max, v_max*1.05)
            ax.set_xlabel('$x$[mm]')
            ax.set_ylabel('$v$[mm/s]')
            ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
            ax.grid(b=True)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Prof+os.sep+'profiles_'+run +'_%06d.png' %LOAD_IDX
            fig.savefig(Name_Out, dpi=65)
            plt.close(fig)
            IMAGES_PROF.append(imageio.imread(Name_Out))
        
        # plot the flux as a function of y
        if PLOT_FLUX == True:
            fig, ax = plt.subplots(figsize=(9, 5))
            # integrate using trapz
            q = np.trapz(v_pad, x_pad)/Scale
            ax.set_title('$t$ = %03d [ms]' %(t[j]*1000))
            ax.scatter(y[:,0], q, c='r', marker='x', s=(300./fig.dpi)**2)
            ax.plot(y[:,0], q, c='r')
            ax.set_ylim(-800,1200)
            ax.set_yticks(np.linspace(-800,1200,11))
            ax.set_xlim(0, Height)
            ax.set_xticks(np.arange(0,Height,4*Scale))
            ax.set_xticklabels(np.linspace(0,20,6,dtype=int), fontsize=15)
            ax.set_xlabel('$y$[mm]', fontsize = 20)
            ax.set_ylabel('$q$[mm$^2$/s]', fontsize = 20)
            ax.grid(b=True)
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Flux+os.sep+'flux%06d.png'%LOAD_IDX
            fig.savefig(Name_Out, dpi=65)
            plt.close(fig)
            IMAGES_FLUX.append(imageio.imread(Name_Out))        
        
    if PLOT_ROI == True:
        imageio.mimsave(GIF_ROI, IMAGES_ROI, duration = duration)
    if PLOT_PROF == True:
        imageio.mimsave(GIF_PROF, IMAGES_PROF, duration= duration)
    if PLOT_H == True:
        imageio.mimsave(GIF_H, IMAGES_H, duration = duration)
    # shutil.rmtree(Fol_Gif_Images)