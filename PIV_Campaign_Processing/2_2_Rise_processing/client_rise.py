# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:42:13 2019

@author: Manuel Ratz
@description: Code to process all the Rises from the
    experimental PIV campaign on the channel. Loaded
    functions are specific to this very case, so care
    about using them
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
settings.sig2noise_method = 'peak2RMS'
settings.sig2noise_mask = 3
settings.do_sig2noise_validation = True 
settings.sig2noise_threshold = 6.5
# validation
settings.validation_first_pass = True
settings.MinMax_U_disp = (-3, 3)
settings.MinMax_V_disp = (-35, 35) 
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
settings.save_plot = False
settings.show_plot = False
settings.scale_plot = 1
settings.plot_ROI = False

'Image preprocessing, is not done'
settings.dynamic_masking_method = 'None'
settings.dynamic_masking_threshold = 0.005
settings.dynamic_masking_filter_size = 7 

# here we load the file containing the beginning index for every run
observation_periods = np.genfromtxt('observation_rise.txt', dtype=str)

# Pietro: this is the index of the test case that I have send to you
run = 22

for i in range(run, run+1):
    # set the folder in which the raw images are located
    settings.filepath_images = 'C:\PIV_Processed\Images_Preprocessed'+os.sep+observation_periods[i,0]
    # set the name to the output foler
    settings.save_folder_suffix = observation_periods[i, 0]
    # sest the image sequence and the beginning index
    settings.frame_pattern_a = observation_periods[i, 0] + '.*.tif'
    settings.frame_pattern_b = None  
    settings.beginning_index = int(observation_periods[i, 1])
    settings.init_ROI = int(observation_periods[i, 2])
    piv(settings)