# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:46:08 2020

@author: manue
@description: Create animations for all the falls (except F_h1_1200_1_q). Here
    we load all the smoothed velocity profiles and animate different things for 
    the large
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
Settings = '_24_24'
Data_Location = 'C:\PIV_Processed' + os.sep
Run_List = os.listdir(Data_Location + 'Images_Processed' + os.sep + 'Fall' + Settings + '_peak2RMS')
# create a postprocessing folder to store the results
Fol_PP = ppf.create_folder('C:\PIV_Processed\Images_Postprocessed' + Settings + os.sep)
Periods = np.genfromtxt('observation_fall.txt', dtype=int)

##############################################################################
######## enable or disable the creation of individual gifs ###################
##############################################################################

PLOT_ROI = True # working
PLOT_PROF = True # working
PLOT_FLUX = True # working
PLOT_S2N = True # working
PLOT_HP = True # working
PLOT_QUIV = True # working

##############################################################################
################ set some cosmetics ##########################################
##############################################################################

Stp_T = 2 # step size in time
Duration = 0.05 # duration of each image in the gif
Dpi = 65

##############################################################################
################ iterate over all the runs ###################################
##############################################################################

# for i in range(0,len(Run_List)):
for i in range(5, len(Run_List)):
# for i in range(1,5):
    Run = ppf.cut_processed_name(Run_List[i])
    print('Exporting Run ' + Run)
    # give the input folder for the data
    Fol_Out = ppf.create_folder(Fol_PP + Run + os.sep)
    Fol_In = Data_Location + 'Images_Processed' + os.sep + 'Fall' + Settings +\
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
    Dt = 1/ppf.get_frequency(Fol_Raw) # time between images
    Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s
    
    # load the data of the predicted height
    H = ppf.load_h(Fol_In)
    Idx1 = Periods[i,1]
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
    GIF_FLUX = Fol_Out + 'Flux' + Gif_Suffix
    GIF_S2N = Fol_Out + 's2n' + Gif_Suffix
    GIF_QUIV = Fol_Out + 'Quiver' + Gif_Suffix
    GIF_HP = Fol_Out + 'Highpass' + Gif_Suffix
    
    # create a folder to store the images
    Fol_Gif_ROI = ppf.create_folder(Fol_Out + os.sep + 'Images_ROI_' + Run + os.sep)
    Fol_Gif_Prof = ppf.create_folder(Fol_Out + os.sep + 'Images_Prof_' + Run + os.sep)
    Fol_Gif_Flux = ppf.create_folder(Fol_Out + os.sep + 'Images_Flux_' + Run + os.sep)
    Fol_Gif_s2n = ppf.create_folder(Fol_Out + os.sep + 'Images_s2n_' + Run + os.sep)
    Fol_Gif_Quiv = ppf.create_folder(Fol_Out + os.sep + 'Images_Quiver_' + Run + os.sep)
    Fol_Gif_HP = ppf.create_folder(Fol_Out + os.sep + 'Images_Highpass_' + Run + os.sep)
    
    # create empty lists to append the images into
    IMAGES_ROI = []
    IMAGES_PROF = []
    IMAGES_FLUX = []
    IMAGES_S2N = []
    IMAGES_QUIV = []
    IMAGES_HP = []
    
    # get the axis parameters
    flux_max, flux_min, flux_ticks = ppf.flux_parameters(v_tensor_smo, x_tensor[0,-1,:], Scale, case = 'Fall')
    flux = ppf.calc_flux(x_tensor[0,-1,:], v_tensor_smo, Scale)
    vmax, vmin, vticks, vticks_expanded = ppf.profile_parameters(v_tensor_smo, case = 'Fall')
    hmax_mm, h_ticks_mm = ppf.height_parameters(h_mm)
    v_max_quiv = np.nanmax(np.abs(v_tensor_smo))
    y_spacing = y_tensor[0,-2,0] - y_tensor[0,-1,0]
    
    v_avg_max = ppf.get_v_avg_max(v_tensor_smo, x_tensor, Width, case = 'Fall')
    acc_max = ppf.get_acc_max(v_tensor_smo, x_tensor, Dt, Width, case = 'Fall')
    
    if Run[::-1].startswith('q'):
        hp_scale = 750
    else:
        hp_scale = 300
    # iterate over all the iamges in the current run
    for j in range(0, N_T):
        # print('Image %d of %d' %((j+1,N_T))) # update the user
        # calculate the loading index and load s2n ratio and the raw image
        Load_Idx = Stp_T*j+0
        Image_Idx = Load_Idx + Frame0
        s2n = ppf.load_s2n(Fol_In, Image_Idx)
        img = ppf.load_raw_image(Fol_Raw, Image_Idx)
        valid_row = ppf.first_valid_row(u_tensor[Load_Idx,:,:])
        
        # extract the current timestep from the 3d tensors
        x_pad = x_tensor[Load_Idx,valid_row:,:]
        y_pad = y_tensor[Load_Idx,valid_row:,:]
        u_pad = u_tensor[Load_Idx,valid_row:,:]
        v_pad = v_tensor_smo[Load_Idx,valid_row:,:]
        # calculate the shifted
        x_pco, y_pco = ppf.shift_grid(x_pad, y_pad, padded = True)
        x = x_pad[:,1:-1]
        y = y_pad[:,1:-1]
        u = u_pad[:,1:-1]
        v = v_pad[:,1:-1]
        
        # plot the s2n histogram
        if PLOT_S2N == True:
            fig, ax = plt.subplots(figsize = (6.9, 4.5))
            ax.hist(s2n.ravel(), bins = 100, density = True)
            ax.set_xlim(0, 25)
            ax.set_ylim(0, 0.3)
            ax.grid(b = True)
            ax.set_xlabel('Signal to noise ratio')
            ax.set_ylabel('Probability')
            ax.plot((6.5, 6.5), (0, 0.3), c = 'orange', label = 'Threshold')
            ax.legend(prop={'size': 15}, loc = 'upper right')
            Name_Out = Fol_Gif_s2n+os.sep+'s2n%06d.png'% Image_Idx
            fig.tight_layout(pad=0.5)
            fig.savefig(Name_Out, dpi = Dpi)
            plt.close(fig)
            IMAGES_S2N.append(imageio.imread(Name_Out)) # append into list
        
        # plot the changing ROI on the Raw Image
        if PLOT_ROI == True:
            # create the figure
            fig, ax = plt.subplots(figsize = (2.5,7.5))
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
            if h_px[j] > 0:
                interface_line = np.ones((img.shape[1],1))*(Height-h_px[j])
                ax.plot(interface_line, lw = 1, c='red')
            Name_Out = Fol_Gif_ROI+os.sep+'roi%06d.png' % Image_Idx # set the output name
            fig.savefig(Name_Out, dpi = Dpi) # save the figure
            plt.close(fig) # close to not overcrowd
            IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
         
        # plots of the velocity profiles    
        if PLOT_PROF == True:
            fig, ax = plt.subplots(figsize=(6.9, 4.5))
            # initialize array with values every 25% of the ROI
            Y_IND = np.array([int((x_pad.shape[0]*0.25)-1),int((x_pad.shape[0]*0.5)-1),int((x_pad.shape[0]*0.75)-1)]) 
            # ax.set_title(ppf.separate_name(Run))
            ax.plot(x_pad[Y_IND[0],:], v_pad[Y_IND[0],:], c='r', label='75\% ROI')
            ax.plot(x_pad[Y_IND[1],:], v_pad[Y_IND[1],:], c='b', label='50\% ROI')
            ax.plot(x_pad[Y_IND[2],:], v_pad[Y_IND[2],:], c='g', label='25\% ROI')
            ax.set_xlim(0, Width-1)
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel('$x$[mm]')
            ax.set_ylabel('$v$[mm/s]')
            ax.legend(prop={'size': 12}, ncol = 3, loc='lower center')
            ax.grid(b=True)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            ax.set_yticks(vticks_expanded)
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Prof+os.sep+'profiles%06d.png' % Image_Idx
            fig.savefig(Name_Out, dpi = Dpi)
            plt.close(fig)
            IMAGES_PROF.append(imageio.imread(Name_Out))
        
        if PLOT_FLUX == True:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            # integrate using trapz
            flux = np.trapz(v_pad, x_pad)/Scale
            ax.scatter(y_pad[:,0], flux, c='r', marker='x', s=(300./fig.dpi)**2)
            ax.plot(y_pad[:,0], flux, c='r')
            ax.set_ylim(flux_min,flux_max)
            ax.set_yticks(flux_ticks)
            ax.set_xlim(0, Height)
            ax.set_xticks(y_ticks)
            ax.set_xticklabels(y_ticklabels)
            ax.set_xlabel('$y$[mm]')
            ax.set_ylabel('$q$[mm$^2$/s]')
            ax.grid(b=True)
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Flux+os.sep+'flux%06d.png' % Image_Idx
            fig.savefig(Name_Out, dpi = Dpi)
            plt.close(fig)
            IMAGES_FLUX.append(imageio.imread(Name_Out))   
        
        # plot the flux as a function of y
        if PLOT_QUIV == True:     
            fig, ax = plt.subplots(figsize = (3.8, 7.58))
            cs = plt.pcolormesh(x_pco, y_pco, v, vmin=vmin, vmax=vmax, cmap = plt.cm.viridis, alpha = 1) # create the contourplot using pcolormesh
            ax.set_aspect('equal') # set the correct aspect ratio
            clb = fig.colorbar(cs, pad = 0.1) # get the colorbar
            clb.set_ticks(vticks) # set the colorbarticks
            ticklabs = clb.ax.get_yticklabels()
            clb.ax.set_yticklabels(ticklabs)
            clb.set_label('Velocity [mm/s]') # set the colorbar title
            clb.set_alpha(1)
            clb.draw_all()
            STEPY= 4
            STEPX = 1
            plt.quiver(x[(x.shape[0]+1)%2::STEPY, ::STEPX], y[(x.shape[0]+1)%2::STEPY, ::STEPX],\
                       u[(x.shape[0]+1)%2::STEPY, ::STEPX], v[(x.shape[0]+1)%2::STEPY, ::STEPX],\
                       color='k', units = 'width', scale_units = 'y', width = 0.005, scale = 1.1*v_max_quiv/y_spacing/STEPY,\
                       )
            ax.set_ylim(0, Height)
            ax.set_xlim(0, Width-1)
            ax.set_yticks(y_ticks) # set custom y ticks
            ax.set_yticklabels(y_ticklabels) # set custom y ticklabels
            ax.set_xticks(x_ticks) # set custom x ticks 
            ax.set_xticklabels(x_ticklabels) # set custom x ticklabels
            ax.set_xlabel('$x$[mm]')
            # ax.set_ylabel('$y$[mm]')
            fig.tight_layout(pad=0.5)
            Name_Out = Fol_Gif_Quiv + os.sep + 'quiver%06d.png'% Image_Idx
            fig.savefig(Name_Out, dpi = Dpi)
            plt.close(fig)
            IMAGES_QUIV.append(imageio.imread(Name_Out))

        if PLOT_HP == True:
            qfield = ppf.calc_qfield(x, y, u, v)
            x_pco, y_pco = ppf.shift_grid(x, y, padded = False)
            fig, ax = plt.subplots(figsize = (3.8, 7.58))
            cs = plt.pcolormesh(x_pco, y_pco, qfield, cmap = plt.cm.viridis, alpha = 1, vmin = -0.1, vmax = 0.06) # create the contourplot using pcolormesh
            ax.set_aspect('equal') # set the correct aspect ratio
            clb = fig.colorbar(cs, pad = 0.1, drawedges = False, alpha = 1) # get the colorbar
            clb.set_label('Q Field [1/s$^2$]')
            ax.set_xlim(0,Width-1)
            ax.set_ylim(0,Height)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels)
            ax.set_xlabel('$x$[mm]')
            # ax.set_ylabel('$y$[mm]')
            fig.tight_layout(pad=0.5)
            u_hp, v_hp = ppf.high_pass(u, v, 3, 3, padded = False)
            StpX = 1
            StpY = 1
            ax.grid(b = False)
            ax.quiver(x[(x.shape[0]+1)%2::StpY,::StpX], y[(x.shape[0]+1)%2::StpY,::StpX],\
                      u_hp[(x.shape[0]+1)%2::StpY,::StpX], v_hp[(x.shape[0]+1)%2::StpY,::StpX],\
                      scale = hp_scale, color = 'k', scale_units = 'height', units = 'width', width = 0.005)
                # scale = 300
            Name_Out = Fol_Gif_HP + os.sep + 'test%06d.png' % Image_Idx
            fig.savefig(Name_Out, dpi = Dpi)
            plt.close(fig)
            IMAGES_HP.append(imageio.imread(Name_Out))
        
    # render the gifs if desired
    if PLOT_ROI == True:
        imageio.mimsave(GIF_ROI, IMAGES_ROI, duration = Duration)
    if PLOT_PROF == True:
        imageio.mimsave(GIF_PROF, IMAGES_PROF, duration= Duration)
    if PLOT_FLUX == True:
        imageio.mimsave(GIF_FLUX, IMAGES_FLUX, duration = Duration)
    if PLOT_S2N == True:
        imageio.mimsave(GIF_S2N, IMAGES_S2N, duration = Duration)
    if PLOT_QUIV == True:
        imageio.mimsave(GIF_QUIV, IMAGES_QUIV, duration = Duration)
    if PLOT_HP == True:
        imageio.mimsave(GIF_HP, IMAGES_HP, duration = Duration)
    
    ppf.save_run_params(Fol_Out + 'Params_' + Run + '.txt', Run, v_avg_max, t[-1], Frame0, NX, 1/Dt, Scale, Height, Width, acc_max)