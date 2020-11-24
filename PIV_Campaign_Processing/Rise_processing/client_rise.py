# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:42:13 2019

@author: Manuel
@description
##############################################################################

 IMPORTANT READ ME           IMPORTANT READ ME              IMPORTANT READ ME
                               
##############################################################################

This code contains the power to process all the rises of the piv campaign. To
use it simply clone the github repository and you will have all the files. In
the end there is a loop. To process all the images simply let the file run as is.

This will likely result in an error. You have to restart the kernel after a few
runs depending on the power of your computer. To be sure restart after every run.

The idea is to run two different settings:
    One with rectangles (Do for ALL images), the window sizes for this are 
        settings.window_height = (256, 128, 64)
        settings.overlap_height = (128, 64, 32)
        settings.window_width = (64, 32, 16)
        settings.overlap_width = (32, 16, 8) 
    One with squares (Do for the 1000 and 1200 Hz runs), the window sizes for this are
        settings.window_height = (128, 64, 32)
        settings.overlap_height = (64, 32, 16)
        settings.window_width = (128, 64, 32)
        settings.overlap_width = (64, 32, 16)

The raw images are stored in a .rar file under the following link (YET TO BE DONE, 24.11.)
    https://osf.io/vp3cq/
Simply download them and store them in the given data path or change according to your needs.

Currently we are not processing two of the rises:
    - R_h1_f750_p10 - because the displacement is too high
    - R_h1_f1000_1_p14 - because two images have been replaced by dummys which
                         will mess up the ROI prediction
                         
Except for the settings in the loop, no other ones should be changed.
"""

import os
import numpy as np
from windef_rect_rise import piv

class Settings(object):
    pass  
settings = Settings()
# Folder for the outputs
settings.save_path = 'C:\PIV_Processed\Images_Processed' + os.sep

# windows and displacement calculation
settings.interpolation_order = 3
settings.subpixel_method = 'gaussian'
settings.correlation_method = 'circular'  # 'circular' because it is faster
settings.iterations = 3 

settings.window_height = (256, 128, 64)
settings.overlap_height = (128, 64, 32)
settings.window_width = (64, 32, 16)
settings.overlap_width = (32, 16, 8) 
# sig2noise
settings.extract_sig2noise = True
settings.sig2noise_method = 'peak2peak'
settings.sig2noise_mask = 3
settings.do_sig2noise_validation = True 
settings.sig2noise_threshold = 1.3
# validation
settings.validation_first_pass = True
settings.MinMax_U_disp = (-3, 3)
settings.MinMax_V_disp = (-35, 35) # this has to be that large because of the 750 Hz runs
settings.std_threshold = 70 # threshold of the std validation, filter disabled
settings.median_threshold = 50  # threshold of the median validation, filter disabled
settings.median_size = 1 
settings.replace_vectors = True 
settings.filter_method = 'localmean'
settings.max_filter_iteration = 4
settings.filter_kernel_size = 1 
# smoothing
settings.smoothn=False
settings.smoothn_p=0.01
# cosmetics; set the rescaling to one because we can do all of this in the post processing
settings.scaling_factor = 1 
settings.dt = 1
settings.save_plot = True
settings.show_plot = False
settings.scale_plot = 1
settings.plot_ROI = False

'Image preprocessing, is not done'
settings.dynamic_masking_method = 'None'
settings.dynamic_masking_threshold = 0.005
settings.dynamic_masking_filter_size = 7 

# here we load the file containing the beginning index for every run
observation_periods = np.genfromtxt('observation_rise.txt', dtype=str)

# iterate over all cases
for i in range(0, len(observation_periods)):
    # set the folder in which the raw images are located
    settings.filepath_images = 'C:\PIV_Processed\Images_Preprocessed'+os.sep+observation_periods[i,0]
    # set the name to the output foler
    settings.save_folder_suffix = observation_periods[i, 0]
    # sest the image sequence and the beginning index
    settings.frame_pattern_a = observation_periods[i, 0] + '.*.tif'
    settings.frame_pattern_b = None  
    settings.beginning_index = int(observation_periods[i, 1])
    piv(settings)















