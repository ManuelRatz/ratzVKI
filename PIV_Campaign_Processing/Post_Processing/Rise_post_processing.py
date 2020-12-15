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
import imageio                          # for animations

# set the plot parameters
ppf.set_plot_parameters(20, 15, 10)

# here a list of all the runs that will be processed is created
Settings = '_64_16'
Data_Location = 'C:\PIV_Processed' + os.sep
Run_List = os.listdir(Data_Location + 'Images_Processed' + os.sep + 'Rise' + Settings + '_peak2RMS')
# create a postprocessing folder to store the results
Fol_PP = ppf.create_folder('C:\PIV_Processed\Images_Postprocessed' + Settings + os.sep)

##############################################################################
######## enable or disable the creation of individual gifs ###################
##############################################################################

PLOT_ROI = False
PLOT_PROF = False
PLOT_H = False
PLOT_FLUX = True

##############################################################################
################ set some cosmetics ##########################################
##############################################################################

Stp_T = 8 # step size in time
Duration = 0.05 # duration of each image in the gif


##############################################################################
################ iterate over all the runs ###################################
##############################################################################

# for i in range(0,len(run_list)):
for i in range(0,1):
    Run = ppf.cut_processed_name(Run_List[i])
    print('Exporting Run ' + Run)
    # give the input folder for the data
    Fol_Out = ppf.create_folder(Fol_PP + Run + os.sep)
    Fol_In = Data_Location + 'Images_Processed' + os.sep + 'Rise' + Settings +\
        '_peak2RMS'+ os.sep + 'Results_' + Run + Settings
    Fol_Raw = ppf.get_raw_folder(Fol_In)
    Fol_Smo = ppf.get_smo_folder(Fol_In)
    
    
    x_tensor = np.load(os.path.join(Fol_Smo, 'x_values.npy'))
    y_tensor = np.load(os.path.join(Fol_Smo, 'y_values.npy'))
    u_tensor = np.load(os.path.join(Fol_Smo, 'u_values.npy'))
    v_tensor_raw = np.load(os.path.join(Fol_Smo, 'v_values_raw.npy'))
    v_tensor_smo = np.load(os.path.join(Fol_Smo, 'v_values_smoothed.npy'))
    
    # set the constants of the images to go from px/frame to mm/s
    Height, Width = ppf.get_img_shape(Fol_Raw) # image width in pixels
    Scale = Width/5 # scaling factor in px/mm
    Dt = 1/ppf.get_frequency(Fol_Raw) # time between images
    Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s
    
    # load the data of the predicted height
    h = ppf.load_h(Fol_In)
    Idx1 = np.nanargmax(h>0)
    Idx2 = len(h)-np.nanargmax(h[::-1]!=0)
    # use a dummy to calculate the indices of valid images (that were processed)
    H_time = h[Idx1:Idx2]
    # set up a time array for plotting
    t = np.linspace(0,len(H_time)*Dt,len(H_time))
    
    # calculate the number of columns and some other constants
    NX = ppf.get_column_amount(Fol_In) # get the number of columns
    Frame0 = Idx1+1 # starting index of the run
    Seconds = (Idx2-Idx1)*Dt # how many seconds to observe the whole thing
    N_T = int((Seconds/Dt)/Stp_T)
    
    # these are the ticks and ticklabels to go from pixel -> mm for the coordinates
    y_ticks = np.arange(0,Height,4*Scale)
    y_ticklabels = np.arange(0, 4*(Height/Width+1), 4, dtype = int)
    x_ticks = np.linspace(0,Width-1, 6)
    x_ticklabels = np.arange(0,6,1)
    
    # create the gif names
    Gif_Suffix = '_' + Run + '.gif'
    GIF_ROI = Fol_Out + 'Changing_Roi' + Gif_Suffix
    GIF_PROF = Fol_Out + 'Profiles' + Gif_Suffix
    GIF_H = Fol_Out + 'H_of_t' + Gif_Suffix
    GIF_FLUX = Fol_Out + 'Flux' + Gif_Suffix
    
    # create a folder to store the images
    Fol_Gif_ROI = ppf.create_folder(Fol_Out + os.sep + 'Images_ROI_' + Run + os.sep)
    Fol_Gif_Hist = ppf.create_folder(Fol_Out + os.sep + 'Images_Hist_' + Run + os.sep)
    Fol_Gif_H = ppf.create_folder(Fol_Out + os.sep + 'Images_h_' + Run + os.sep)
    Fol_Gif_Prof = ppf.create_folder(Fol_Out + os.sep + 'Images_Prof_' + Run + os.sep)
    Fol_Gif_Flux = ppf.create_folder(Fol_Out + os.sep + 'Images_Flux_' + Run + os.sep)
    
    # create empty lists to append the images into
    IMAGES_ROI = []
    IMAGES_PROF = []
    IMAGES_H = []
    IMAGES_FLUX = []
    
    # get the axis parameters
    flux_max, flux_min, flux_ticks = ppf.flux_parameters(v_tensor_smo, x_tensor[0,-1,:], Scale)
    
    # iterate over all the iamges in the current run
    N_T = 1
    # calculate the loading index and load the data
    for j in range(0, N_T):
        print('Image %d of %d' %((j+1,N_T))) # update the user
        # calculate the loading index and load the data
        Load_Idx = Stp_T*j
        Image_Idx = Load_Idx + Frame0
        # x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, Load_Idx, NX)
        # # filter out the invalid rows
        # # x, y, u, v, ratio, valid, invalid = ppf.filter_invalid(x, y, u, v, ratio, mask, valid_thresh = 0.5)    
        # # convert to mm/s
        # u = u*Factor
        # v = v*Factor
        # # pad the data using the no slip boundary condition
        # x_pad, y_pad, u_pad, v_pad = ppf.pad(x, y, u, v, Width)   
        # x_pco, y_pco = ppf.shift_grid(x, y)
        # u_hp, v_hp = ppf.high_pass(u, v, sigma = 3, truncate = 3)
        # img = ppf.load_raw_image(Fol_Raw, Load_Idx)
        
        x_pad = x_tensor[Load_Idx,:,:]
        y_pad = y_tensor[Load_Idx,:,:]
        u_pad = u_tensor[Load_Idx,:,:]
        v_pad = v_tensor_smo[Load_Idx,:,:]
        
        # # plots of the changing ROI
        # if PLOT_ROI == True:
        #     # create the figure
        #     fig, ax = plt.subplots(figsize = (2.5,8))
        #     ax.set_ylim(0,Height)
        #     ax.set_xlim(0,Width)
        #     ax.imshow(img, cmap=plt.cm.gray) # show the image
        #     ax.set_yticks(y_ticks) # set custom y ticks
        #     ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
        #     ax.set_xticks(x_ticks) # set custom x ticks 
        #     ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
        #     ax.set_xlabel('$x$[mm]') # set x label
        #     ax.set_ylabel('$y$[mm]') # set y label
        #     fig.tight_layout(pad=0.5) # crop edges of the figure to save space
        #     # plot a horizontal line of the predicted interface in case it is visible
        #     if h[Load_Idx] > 0:
        #         interface_line = np.ones((img.shape[1],1))*(Height-h[Load_Idx])
        #         ax.plot(interface_line, lw = 1, c='r')
        #     Name_Out = Fol_Gif_ROI+os.sep+'roi%06d.png' % Image_Idx # set the output name
        #     fig.savefig(Name_Out, dpi = 80) # save the figure
        #     plt.close(fig) # close to not overcrowd
        #     IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
            
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
            Name_Out = Fol_Gif_H + os.sep+'h%06d.png' % Image_Idx
            fig.savefig(Name_Out, dpi=80)
            plt.close(fig)
            IMAGES_H.append(imageio.imread(Name_Out))
         
        # # plots of the velocity profiles    
        # if PLOT_PROF == True:
        #     fig, ax = plt.subplots(figsize=(8,5))
        #     # initialize array with values every 25% of the ROI
        #     Y_IND = np.array([int(len(x)*0.25)-1,int(len(x)*0.5)-1,int(len(x)*0.75)-1]) 
        #     ax.set_title(ppf.separate_name(Run))
        #     ax.plot(x_pad[Y_IND[0],:], v_pad[Y_IND[0],:], c='r', label='75\% ROI')
        #     ax.plot(x_pad[Y_IND[1],:], v_pad[Y_IND[1],:], c='b', label='50\% ROI')
        #     ax.plot(x_pad[Y_IND[2],:], v_pad[Y_IND[2],:], c='g', label='25\% ROI')
        #     ax.set_xlim(0, Width)
        #     ax.set_ylim(-0.6*v_max, v_max*1.05)
        #     ax.set_xlabel('$x$[mm]')
        #     ax.set_ylabel('$v$[mm/s]')
        #     ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
        #     ax.grid(b=True)
        #     ax.set_xticks(x_ticks)
        #     ax.set_xticklabels(x_ticklabels)
        #     fig.tight_layout(pad=0.5)
        #     Name_Out = Fol_Gif_Prof+os.sep+'profiles%06d.png' % Image_Idx
        #     fig.savefig(Name_Out, dpi=80)
        #     plt.close(fig)
        #     IMAGES_PROF.append(imageio.imread(Name_Out))
        
        # plot the flux as a function of y
        if PLOT_FLUX == True:
            fig, ax = plt.subplots(figsize=(9, 5))
            # integrate using trapz
            q = np.trapz(v_pad, x_pad)/Scale
            ax.set_title('$t$ = %03d [ms]' %(t[j]*1000))
            ax.scatter(y_pad[:,0], q, c='r', marker='x', s=(300./fig.dpi)**2)
            ax.plot(y_pad[:,0], q, c='r')
            ax.set_ylim(flux_min,flux_max)
            ax.set_yticks(flux_ticks)
            ax.set_xlim(0, Height)
            ax.set_xticks(y_ticks)
            ax.set_xticklabels(y_ticklabels)
            ax.set_xlabel('$y$[mm]', fontsize = 20)
            ax.set_ylabel('$q$[mm$^2$/s]', fontsize = 20)
            ax.grid(b=True)
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Flux+os.sep+'flux%06d.png' % Image_Idx
            fig.savefig(Name_Out, dpi=80)
            plt.close(fig)
            IMAGES_FLUX.append(imageio.imread(Name_Out))        
        
    if PLOT_ROI == True:
        imageio.mimsave(GIF_ROI, IMAGES_ROI, duration = Duration)
    if PLOT_PROF == True:
        imageio.mimsave(GIF_PROF, IMAGES_PROF, duration= Duration)
    if PLOT_H == True:
        imageio.mimsave(GIF_H, IMAGES_H, duration = Duration)
    if PLOT_FLUX == True:
        imageio.mimsave(GIF_FLUX, IMAGES_FLUX, duration = Duration)