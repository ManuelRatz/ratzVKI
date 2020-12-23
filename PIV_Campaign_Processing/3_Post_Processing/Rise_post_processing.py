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

PLOT_ROI = False # working
PLOT_PROF = False # working
PLOT_H = False # working
PLOT_FLUX = False # working 
PLOT_S2N = False # working
PLOT_HP = False # working
PLOT_QUIV = False # working

##############################################################################
################ set some cosmetics ##########################################
##############################################################################

# Stp_T = 800 # step size in time
Duration = 0.05 # duration of each image in the gif
Dpi = 65 # dpi of the figures

##############################################################################
################ iterate over all the runs ###################################
##############################################################################

for i in range(0,len(Run_List)):
# for i in range(13, 17):
    Run = ppf.cut_processed_name(Run_List[i]) # current run
    print('Exporting Run ' + Run)
    # give the input folder for the data
    Fol_Out = ppf.create_folder(Fol_PP + Run + os.sep)
    Fol_In = Data_Location + 'Images_Processed' + os.sep + 'Rise' + Settings +\
        '_peak2RMS'+ os.sep + 'Results_' + Run + Settings
    Fol_Raw = ppf.get_raw_folder(Fol_In)
    Fol_Smo = ppf.get_smo_folder(Fol_In)
    
    # load the smoothed data from the .npy files
    x_tensor = np.load(os.path.join(Fol_Smo, 'x_values.npy'))
    y_tensor = np.load(os.path.join(Fol_Smo, 'y_values.npy'))
    u_tensor = np.load(os.path.join(Fol_Smo, 'u_values.npy'))
    v_tensor_raw = np.load(os.path.join(Fol_Smo, 'v_values_raw.npy'))
    v_tensor_smo = np.load(os.path.join(Fol_Smo, 'v_values_smoothed.npy'))
    
    # set the constants of the images to go from px/frame to mm/s
    Height, Width = ppf.get_img_shape(Fol_Raw) # image width in pixels
    Scale = Width/5 # scaling factor in px/mm
    Freq = ppf.get_frequency(Fol_Raw) # acquisition frequency of the images 
    Dt = 1/Freq # time between images
    Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s
    
    # different time steps for low acquisition frequency
    if Freq == 750:
        Stp_T = 5
    else:
        Stp_T = 8
    
    # load the data of the predicted height
    H = ppf.load_h(Fol_In)
    Idx1 = np.nanargmax(H>0)
    Idx2 = len(H)-np.nanargmax(H[::-1]!=0)
    # use a dummy to calculate the indices of valid images (that were processed)
    H_time = H[Idx1:Idx2]
    # calculate the number of columns, Frame0, the duration and the number of images of the run
    NX = ppf.get_column_amount(Fol_In) # get the number of columns
    Frame0 = Idx1+1 # starting index of the run
    Seconds = (Idx2-Idx1)*Dt # how many seconds to observe the whole thing
    N_T = int((Seconds/Dt)/Stp_T)
    # crop the height before frame0
    h_px = H[Frame0-1:]
    # shift it to only get the steps
    h_px = h_px[::Stp_T]
    # calculate the height in mm
    h_mm = (-h_px+Height)/Scale
    # set up a time array for plotting
    t = np.linspace(0,len(H_time)*Dt,len(H_time))
    # shift it as well
    t = t[::Stp_T]
    
    # these are the ticks and ticklabels to go from pixel -> mm for the coordinates
    # x is with respect to the horizontal axis and y with respect to the vertical one
    # depending on the plot, sometimes the y values are on the horizontal axis
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
    GIF_S2N = Fol_Out + 's2n' + Gif_Suffix
    GIF_QUIV = Fol_Out + 'Quiver' + Gif_Suffix
    GIF_HP = Fol_Out + 'Highpass' + Gif_Suffix
    
    # create a folder to store the images
    Fol_Gif_ROI = ppf.create_folder(Fol_Out + os.sep + 'Images_ROI_' + Run + os.sep)
    Fol_Gif_H = ppf.create_folder(Fol_Out + os.sep + 'Images_h_' + Run + os.sep)
    Fol_Gif_Prof = ppf.create_folder(Fol_Out + os.sep + 'Images_Prof_' + Run + os.sep)
    Fol_Gif_Flux = ppf.create_folder(Fol_Out + os.sep + 'Images_Flux_' + Run + os.sep)
    Fol_Gif_s2n = ppf.create_folder(Fol_Out + os.sep + 'Images_s2n_' + Run + os.sep)
    Fol_Gif_Quiv = ppf.create_folder(Fol_Out + os.sep + 'Images_Quiver_' + Run + os.sep)
    Fol_Gif_HP = ppf.create_folder(Fol_Out + os.sep + 'Images_Highpass_' + Run + os.sep)
    
    # create empty lists to append the images into
    IMAGES_ROI = []
    IMAGES_PROF = []
    IMAGES_H = []
    IMAGES_FLUX = []
    IMAGES_S2N = []
    IMAGES_QUIV = []
    IMAGES_HP = []
    
    # get the axis parameters
    flux_max, flux_min, flux_ticks = ppf.flux_parameters(v_tensor_smo, x_tensor[0,-1,:], Scale, case = 'Rise') # max and min flux and ticks
    vmax, vmin, vticks, vticks_expanded = ppf.profile_parameters(v_tensor_smo, case = 'Rise') # max and min velocity and ticks
    hmax_mm, h_ticks_mm = ppf.height_parameters(h_mm) # maximum height and ticks
    v_max_quiv = np.nanmax(v_tensor_smo) # maximum velocity
    
    # vertical distance of two rows for the quiver plot in pixels
    y_spacing = y_tensor[0,-2,0] - y_tensor[0,-1,0] 
    
    # get the maximum mean velocity and acceleration
    v_avg_max = ppf.get_v_avg_max(v_tensor_smo, x_tensor, Width, case = 'Rise')
    acc_max = ppf.get_acc_max(v_tensor_smo, x_tensor, Dt, Width, case = 'Rise')
    
    # iterate over all the iamges in the current run
    for j in range(0, N_T):
        print('Image %d of %d' %((j+1,N_T))) # update the user
        # calculate the loading index and load s2n ratio and the raw image
        Load_Idx = Stp_T*j # load index of the velocity tensor (shifted by Frame0)
        Image_Idx = Load_Idx + Frame0 # load index of the raw image
        s2n = ppf.load_s2n(Fol_In, Image_Idx) # load the s2n ratio
        img = ppf.load_raw_image(Fol_Raw, Image_Idx) # load the image
        valid_row = ppf.first_valid_row(u_tensor[Load_Idx,:,:]) # get the index of the first valid row
        
        # extract the current timestep from the 3d tensors
        x_pad = x_tensor[Load_Idx,valid_row:,:]
        y_pad = y_tensor[Load_Idx,valid_row:,:]
        u_pad = u_tensor[Load_Idx,valid_row:,:]
        v_pad = v_tensor_smo[Load_Idx,valid_row:,:]
        
        # calculate the shifted grid
        x_pco, y_pco = ppf.shift_grid(x_pad, y_pad, padded = True)
        # crop the padded fields to get the 'raw' ones
        x = x_pad[:,1:-1]
        y = y_pad[:,1:-1]
        u = u_pad[:,1:-1]
        v = v_pad[:,1:-1]
        
        # plot the s2n histogram
        if PLOT_S2N == True:
            fig, ax = plt.subplots(figsize = (6.9, 4.5)) # create the figure
            ax.hist(s2n.ravel(), bins = 100, density = True) # plot the histogram
            ax.set_xlim(0, 25) # set x limit
            ax.set_ylim(0, 0.3) # set y limit
            ax.grid(b = True) # enable the grid
            ax.set_xlabel('Signal to noise ratio') # set x label
            ax.set_ylabel('Probability') # set y lable
            ax.plot((6.5, 6.5), (0, 0.3), c = 'orange', label = 'Threshold') # plot the threshold
            ax.legend(prop={'size': 15}, loc = 'upper right') # show the legend
            Name_Out = Fol_Gif_s2n+os.sep+'s2n%06d.png'% Image_Idx # output name
            fig.tight_layout(pad=0.5) # crop figure edges
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close it in the plot window
            IMAGES_S2N.append(imageio.imread(Name_Out)) # append into list
        
        # plot the changing ROI on the Raw Image
        if PLOT_ROI == True:
            fig, ax = plt.subplots(figsize = (2.5,7.5)) # create the figure
            ax.set_ylim(0,Height) # set x limit
            ax.set_xlim(0,Width) # set y limit
            ax.imshow(img, cmap=plt.cm.gray) # show the image
            ax.set_yticks(y_ticks) # set custom y ticks
            ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
            ax.set_xticks(x_ticks) # set custom x ticks 
            ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
            ax.set_xlabel('$x$[mm]') # set x label
            ax.set_ylabel('$y$[mm]') # set y label
            fig.tight_layout(pad=0.5) # crop edges of the figure to save space
            # plot a horizontal line of the predicted interface in case it is visible
            if h_px[j] > 0:
                interface_line = np.ones((img.shape[1],1))*(Height-h_px[j])
                ax.plot(interface_line, lw = 1, c='red')
            Name_Out = Fol_Gif_ROI+os.sep+'roi%06d.png' % Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close it in the plot window
            IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
            
        # plots of the changing height
        if PLOT_H == True:
            fig, ax = plt.subplots(figsize=(6.9,4.5))
            # plot the height with one point at the end
            ax.plot(t[:j+1], h_mm[:j+1], c='red', label = 'Interface Height')
            ax.scatter(t[j], h_mm[j], c='red', marker='x', s=(300./fig.dpi)**2)
            ax.set_ylim(0, hmax_mm) # set y limit
            ax.set_xlim(0, t[-1]) # set x limit
            ax.set_xlabel('$t$[s]') # set x label
            ax.set_ylabel('$h$[mm]') # set y label
            ax.grid(b=True) # enable the grid
            ax.set_yticks(h_ticks_mm) # set the y ticks
            fig.tight_layout(pad=1.1) # crop the figure edges
            ax.legend(prop={'size': 15}, loc='upper right') # show the legend
            Name_Out = Fol_Gif_H + os.sep+'h%06d.png' % Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close it in the plot window
            IMAGES_H.append(imageio.imread(Name_Out)) # append into list
         
        # plots of the velocity profiles    
        if PLOT_PROF == True:
            fig, ax = plt.subplots(figsize=(6.9, 4.5)) # create the figure
            # initialize array with values every 25% of the ROI
            Y_IND = np.array([int((x_pad.shape[0]*0.25)-1),int((x_pad.shape[0]*0.5)-1),int((x_pad.shape[0]*0.75)-1)])
            # plot the three profiles and label them
            ax.plot(x_pad[Y_IND[0],:], v_pad[Y_IND[0],:], c='r', label='75\% ROI')
            ax.plot(x_pad[Y_IND[1],:], v_pad[Y_IND[1],:], c='b', label='50\% ROI')
            ax.plot(x_pad[Y_IND[2],:], v_pad[Y_IND[2],:], c='g', label='25\% ROI')
            ax.set_xlim(0, Width-1) # set x limit
            ax.set_ylim(vmin, vmax) # set y limit
            ax.set_xlabel('$x$[mm]') # set x label
            ax.set_ylabel('$v$[mm/s]') # set y label
            ax.legend(prop={'size': 12}, ncol = 3, loc='lower center') # show the legend
            ax.grid(b=True) # enable the grid
            ax.set_xticks(x_ticks) # set custom x ticks
            ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
            ax.set_yticks(vticks_expanded) # set custom v ticks
            fig.tight_layout(pad=0.5) # crop the figure edges
            Name_Out = Fol_Gif_Prof+os.sep+'profiles%06d.png' % Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close the figure in the plot window
            IMAGES_PROF.append(imageio.imread(Name_Out)) # append into list
        
        if PLOT_FLUX == True:
            fig, ax = plt.subplots(figsize=(7, 4.5)) # create the figure
            # calculate the current flux as a funciton of y
            flux = np.trapz(v_pad, x_pad)/Scale
            # plot the flux with plot and scatter
            ax.scatter(y_pad[:,0], flux, c='r', marker='x', s=(300./fig.dpi)**2)
            ax.plot(y_pad[:,0], flux, c='r')
            ax.set_ylim(flux_min,flux_max) # set y limits
            ax.set_yticks(flux_ticks) # set y ticks
            ax.set_xlim(0, Height) # set x limits
            ax.set_xticks(y_ticks) # set x ticks (this is the vertical image axis, don't be confused)
            ax.set_xticklabels(y_ticklabels) # set custom x ticklabels
            ax.set_xlabel('$y$[mm]') # set x label
            ax.set_ylabel('$q$[mm$^2$/s]') # set y label
            ax.grid(b=True) # enable the grid
            fig.tight_layout(pad=0.5) # crop the figure edges
            Name_Out = Fol_Gif_Flux+os.sep+'flux%06d.png' % Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close it in the plot window
            IMAGES_FLUX.append(imageio.imread(Name_Out)) # append into list   
        
        # plot the flux as a function of y
        if PLOT_QUIV == True:     
            fig, ax = plt.subplots(figsize = (3.8, 7.58)) # create the figure
            # create the contourplot using pcolormesh
            cs = plt.pcolormesh(x_pco, y_pco, v, vmin=vmin, vmax=vmax, cmap = plt.cm.viridis, alpha = 1)
            ax.set_aspect('equal') # equal aspect ratio for proper scaling
            clb = fig.colorbar(cs, pad = 0.1) # get the colorbar
            clb.set_ticks(vticks) # set the colorbarticks
            ticklabs = clb.ax.get_yticklabels() # get the ticklabels of the colorbar
            clb.ax.set_yticklabels(ticklabs) # set them (in case they have to be customized)
            clb.set_label('Velocity [mm/s]') # set the colorbar label
            clb.set_alpha(1) # figure should not be opaque
            clb.draw_all() # in case the alpha is different
            
            STEPY= 4 # step size along the y axis
            STEPX = 1 # step size along the x axis
            # create the quiver plot with custom scaling to not overlap between the rows
            plt.quiver(x[(x.shape[0]+1)%2::STEPY, ::STEPX], y[(x.shape[0]+1)%2::STEPY, ::STEPX],\
                       u[(x.shape[0]+1)%2::STEPY, ::STEPX], v[(x.shape[0]+1)%2::STEPY, ::STEPX],\
                       color='k', units = 'width', scale_units = 'y', width = 0.005,\
                           scale = 1.1*v_max_quiv/y_spacing/STEPY)
            ax.set_ylim(0, Height) # set the y limit
            ax.set_xlim(0, Width-1) # set the x limit
            ax.set_yticks(y_ticks) # set custom y ticks
            ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
            ax.set_xticks(x_ticks) # set custom x ticks 
            ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
            ax.set_xlabel('$x$[mm]') # set the x label
            # ax.set_ylabel('$y$[mm]'), this is not done because they are all
            # shown in one power point
            fig.tight_layout(pad=0.5) # crop the figure edges
            Name_Out = Fol_Gif_Quiv + os.sep + 'quiver%06d.png'% Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close the figure in the plot window
            IMAGES_QUIV.append(imageio.imread(Name_Out)) # append into list
            
        if PLOT_HP == True:
            qfield = ppf.calc_qfield(x, y, u, v) # calculate the qfield
            fig, ax = plt.subplots(figsize = (3.8, 7.58)) # create the figure
            # create the contourplot using pcolormesh
            cs = plt.pcolormesh(x_pco, y_pco, qfield, cmap = plt.cm.viridis, alpha = 1, vmin = -0.1, vmax = 0.06) 
            ax.set_aspect('equal') # set the correct aspect ratio
            clb = fig.colorbar(cs, pad = 0.1, drawedges = False, alpha = 1) # get the colorbar
            clb.set_label('Q Field [1/s$^2$]') # set the colorbar label
            ax.set_xlim(0,Width-1) # set x limit
            ax.set_ylim(0,Height) # set y limit
            ax.set_xticks(x_ticks) # set custom x ticks
            ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
            ax.set_yticks(y_ticks) # set custom y ticks
            ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
            ax.set_xlabel('$x$[mm]') # set x label
            # ax.set_ylabel('$y$[mm]'), this is not done because they are all
            # shown in one power point
            fig.tight_layout(pad=0.5) # crop the figure edges
            u_hp, v_hp = ppf.high_pass(u, v, 3, 3, padded = False) # calculate the highpass filtered velocity field
            StpY= 1 # step size along the y axis
            StpX = 1 # step size along the x axis
            ax.grid(b = False) # disable the grid
            # create the quiver with custom scaling, this is not adaptive 
            # becauses the highpass filtered velocity field is purely for 
            # visualization purposes
            ax.quiver(x[(x.shape[0]+1)%2::StpY,::StpX], y[(x.shape[0]+1)%2::StpY,::StpX],\
                      u_hp[(x.shape[0]+1)%2::StpY,::StpX], v_hp[(x.shape[0]+1)%2::StpY,::StpX],\
                      scale = 300, color = 'k', scale_units = 'height', units = 'width', width = 0.005)
            Name_Out = Fol_Gif_HP + os.sep + 'test%06d.png' % Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close it in the plot window
            IMAGES_HP.append(imageio.imread(Name_Out)) # append into list
        
    # render the gifs if desired
    if PLOT_ROI == True:
        imageio.mimsave(GIF_ROI, IMAGES_ROI, duration = Duration)
    if PLOT_PROF == True:
        imageio.mimsave(GIF_PROF, IMAGES_PROF, duration= Duration)
    if PLOT_H == True:
        imageio.mimsave(GIF_H, IMAGES_H, duration = Duration)
    if PLOT_FLUX == True:
        imageio.mimsave(GIF_FLUX, IMAGES_FLUX, duration = Duration)
    if PLOT_S2N == True:
        imageio.mimsave(GIF_S2N, IMAGES_S2N, duration = Duration)
    if PLOT_QUIV == True:
        imageio.mimsave(GIF_QUIV, IMAGES_QUIV, duration = Duration)
    if PLOT_HP == True:
        imageio.mimsave(GIF_HP, IMAGES_HP, duration = Duration)
    
    ppf.save_run_params(Fol_Out + 'Params_' + Run + '.txt', Run, v_avg_max, t[-1], Frame0, NX, 1/Dt, Scale, Height, Width, acc_max)