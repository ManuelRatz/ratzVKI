# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:42:13 2019

@author: Theo
"""

# add two directories that include the new files
# note that we need to import openpiv in a separate, original namespace
# so we can use everything from openpiv as openpiv.filters and whatever is 
# going to replace it will be just filteers (for example)

import os
from windef_rect_rise import piv

class Settings(object):
    pass  
settings = Settings()

'Data related settings'
settings.save_folder_suffix = 'R_h2_f1200_1_p13'
# Folder with the images to process
settings.filepath_images = 'C:\\Users\manue\Desktop\Rise_test' + os.sep
# Folder for the outputs
settings.save_path = 'C:\\Users\manue\Desktop\Rise_test_processed' + os.sep
# Root name of the output Folder for Result Files

# Format and Image Sequence
settings.frame_pattern_a = settings.save_folder_suffix + '.*.tif'
settings.frame_pattern_b = None    

'Region of interest'
# (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
# settings.ROI = (0,1269,0,500) # The first number is the position of the interface measured from the bottom of the image
settings.ROI = 'full'


'Image preprocessing'
settings.dynamic_masking_method = 'None'
settings.dynamic_masking_threshold = 0.005
settings.dynamic_masking_filter_size = 7 

# windows and displacement calculation
settings.interpolation_order = 3
settings.subpixel_method = 'gaussian'
settings.correlation_method = 'linear'  # 'circular' or 'linear'
settings.iterations = 2 # select the number of PIV passes

"""
Here we set the window sizes. This code uses rectangular windows, if this is 
not desired, simply put the same values for window_height and window_width as
well as overlap_height and overlap_width
"""
# base 2
settings.window_height = (256, 32, 24, 16)
settings.overlap_height = (128, 16, 12, 8)
settings.window_width = (32, 16, 12, 16)
settings.overlap_width = (16, 8, 6, 8) 
# # base 3
# settings.window_height = (192, 96, 48, 24, 12)
# settings.overlap_height = (96, 48, 24, 12, 6) # 50%
# settings.window_width = (48, 24, 12, 6)
# settings.overlap_width = (24, 12, 6, 3) # 50%

# sig2noise
settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
settings.sig2noise_method = 'peak2peak'
settings.sig2noise_mask = 3
settings.do_sig2noise_validation = True # This is time consuming
settings.sig2noise_threshold = 1.3

# validation
settings.validation_first_pass = True
settings.MinMax_U_disp = (-10, 10)
settings.MinMax_V_disp = (-100, 100)
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
settings.save_plot = True
settings.show_plot = False
settings.scale_plot = 200 # select a value to scale the quiver plot of the vectorfield

settings.beginning_index = 0

# run the script with these settings
piv(settings)














