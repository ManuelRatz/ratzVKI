# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:30:17 2020

@author: Manuel Ratz
@description: Code to execute the smoothing for all the runs
    by iterating over them. This is done by using 3d smoothing
    of the velocity tensor. See post_processing_functions.py
    for a detailed explanation of each step
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import post_processing_functions as ppf
import os
import numpy as np

# Fol_Rise = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS'
# Results = ppf.create_folder('C:\PIV_Processed\Fields_Smoothed')
# runs = os.listdir(Fol_Rise)

# for i in range(0, len(runs)):
# # for i in range(0, 1):
#     Name_Cut = ppf.cut_processed_name(runs[i])
#     print(Name_Cut)
#     Fol_Out = ppf.create_folder(os.path.join(Results, 'Smoothed_'+Name_Cut))
#     Directory = os.path.join(Fol_Rise, runs[i])
#     x_tensor, y_tensor, u_tensor, v_tensor_raw, v_tensor_smo = ppf.load_and_smooth(Directory, order = 18, valid_thresh = 0.5)
#     np.save(os.path.join(Fol_Out, 'x_values'), x_tensor)
#     np.save(os.path.join(Fol_Out, 'y_values'), y_tensor)
#     np.save(os.path.join(Fol_Out, 'u_values'), u_tensor)
#     np.save(os.path.join(Fol_Out, 'v_values_raw'), v_tensor_raw)
#     np.save(os.path.join(Fol_Out, 'v_values_smoothed'), v_tensor_smo)
#     print('\n')
    
Fol_Fall = 'C:\PIV_Processed\Images_Processed\Fall_24_24_peak2RMS'
Results = ppf.create_folder('C:\PIV_Processed\Fields_Smoothed')
runs = os.listdir(Fol_Fall)

# for i in range(0, len(runs)):
for i in range(1, len(runs)):
    Name_Cut = ppf.cut_processed_name(runs[i])
    print(Name_Cut)
    Fol_Out = ppf.create_folder(os.path.join(Results, 'Smoothed_'+Name_Cut))
    Directory = os.path.join(Fol_Fall, runs[i])
    x_tensor, y_tensor, u_tensor, v_tensor_raw, v_tensor_smo = ppf.load_and_smooth(Directory, order = 18, valid_thresh = 0.5)
    np.save(os.path.join(Fol_Out, 'x_values'), x_tensor)
    np.save(os.path.join(Fol_Out, 'y_values'), y_tensor)
    np.save(os.path.join(Fol_Out, 'u_values'), u_tensor)
    np.save(os.path.join(Fol_Out, 'v_values_raw'), v_tensor_raw)
    np.save(os.path.join(Fol_Out, 'v_values_smoothed'), v_tensor_smo)
    print('\n')