# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:42:13 2019

@author: Manuel
@description:
##############################################################################

 IMPORTANT READ ME           IMPORTANT READ ME              IMPORTANT READ ME
                               
##############################################################################

This code contains the power to process all the Falls of the piv campaign. To
use it simply clone the github repository and you will have all the files. In
the end there is a loop. To process all the images simply let the file run as is.

This will likely result in an error. You have to restart the kernel after a few
runs depending on the power of your computer. To be sure restart after every run.

The idea is to run two different settings:
    One with rectangles (Do for ALL images), the window sizes for this are 
        settings.window_height = (64, 32, 24)
        settings.overlap_height = (32, 16, 12)
        settings.window_width = (64, 32, 24)
        settings.overlap_width = (32, 16, 12)
    One with squares (Do for the 1000 and 1200 Hz runs), the window sizes for this are
        settings.window_height = (64, 48, 48)
        settings.overlap_height = (32, 24, 24)
        settings.window_width = (48, 24, 12)
        settings.overlap_width = (24, 12, 6) 

The raw images are stored in a .zip file under the following link
    https://osf.io/q6cw4/
Simply download them and store them in the given data path or change according to your needs.

Currently we are not processing two of the rises:
    - R_h1_f750_p10 - because the displacement is too high
    - R_h1_f1000_1_p14 - because two images have been replaced by dummys which
                         will mess up the ROI prediction
                         
Except for the settings in the loop, no other ones should be changed.
"""


import os
import numpy as np
from windef_rect_fall import piv

class Settings(object):
    pass  
settings = Settings()

'Data related settings'

# Folder for the outputs
settings.save_path = 'C:\PIV_Processed\Images_Processed'
settings.ROI = np.array([0,1280,0,500])


'Image preprocessing'
settings.dynamic_masking_method = 'None'
settings.dynamic_masking_threshold = 0.005
settings.dynamic_masking_filter_size = 7 

# windows and displacement calculation
settings.interpolation_order = 3
settings.subpixel_method = 'gaussian'
settings.correlation_method = 'circular'  # 'circular' or 'linear'
settings.iterations = 3 # select the number of PIV passes
# base 2
settings.window_height = (64, 32, 24)
settings.overlap_height = (32, 16, 12)
settings.window_width = (64, 32, 24)
settings.overlap_width = (32, 16, 12)


# sig2noise
settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
settings.sig2noise_method = 'peak2RMS'
settings.sig2noise_mask = 1
settings.do_sig2noise_validation = False # This is time consuming
settings.sig2noise_threshold = 7.5

# validation
settings.validation_first_pass = True
settings.MinMax_U_disp = (-3, 3)
settings.MinMax_V_disp = (-25, 0)
settings.std_threshold = 70 # threshold of the std validation
settings.median_threshold = 50  # threshold of the median validation
settings.median_size = 1 
settings.replace_vectors = True # Enable the replacment. Chosse: True or False
settings.filter_method = 'localmean' # select a method to replace the outliers: 'localmean', 'disk', 'distance'
settings.max_filter_iteration = 4
settings.filter_kernel_size = 1  # kernel size for the localmean method

# smoothing
settings.smoothn=False #Enables smoothing of the displacemenet field
settings.smoothn_p=0.01 # This is a smoothing parameter

# cosmetics
settings.scaling_factor = 1  # scaling factor pixel/meter
settings.dt = 1  # time between to frames (in seconds)
settings.save_plot = False
settings.show_plot = False
settings.scale_plot = 200 # select a value to scale the quiver plot of the vectorfield

observation_periods = np.genfromtxt('observation_fall.txt', dtype=str)

run = 2
for i in range(run, run+1):
    # Folder with the images to process
    settings.filepath_images = 'C:\PIV_Processed\Images_Preprocessed'+os.sep+observation_periods[i,0]
    
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = observation_periods[i, 0] + '_peak2RMS'
    # Format and Image Sequence
    settings.frame_pattern_a = observation_periods[i, 0] + '.*.tif'
    settings.frame_pattern_b = None  
    settings.fall_start = int(observation_periods[i, 1]) # index after which the liquid is moving
    settings.roi_shift_start = int(observation_periods[i, 2]) # index after which to shift the ROI
    settings.process_fall = True
    settings.process_roi_shift = True
    settings.run = 1
    piv(settings)
    















