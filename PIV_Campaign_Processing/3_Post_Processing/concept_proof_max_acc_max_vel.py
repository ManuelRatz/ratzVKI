# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:11:48 2020

@author: Manuel Ratz
@description: Code to proof the concept on how the maximum acceleration and
    velocity are calculated. Includes an example for a Fall
"""
import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from smoothn import smoothn as smo
import post_processing_functions as ppf
# path = 'C:\PIV_Processed\Fields_Smoothed\Smoothed_F_h1_f1000_1_s'
# vel = np.load(os.path.join(path, 'v_values_smoothed.npy'))
# x = np.load(os.path.join(path, 'x_values.npy'))
# Width = x[0,-1,-1]

# v_avg = np.trapz(vel[:,-2,:], x[0,-1,:])/Width


# plt.plot(v_avg)
# v_avg_filter, DUMMY, DUMMY, DUMMY = smo(v_avg, s = 40, isrobust = True)
# v_avg_savgol = savgol_filter(v_avg, 25, 1)
# plt.plot(v_avg_filter)
# plt.plot(v_avg_savgol)
# freq = 1000

# acc = np.gradient(v_avg, 1/freq)
# acc_smo = np.gradient(v_avg_filter, 1/freq)
# acc_sg = np.gradient(v_avg_savgol, 1/freq)

# fig, ax = plt.subplots()
# ax.plot(acc_smo[50:85])
# ax.plot(acc_sg[50:85])
# ax.plot(acc[50:85])

# acc_max_sg = np.nanmax(np.abs(acc_sg))
# acc_max_smo = np.nanmax(np.abs(acc_smo))


path2 = 'C:\PIV_Processed\Fields_Smoothed\Smoothed_R_h1_f1200_1_p14'
v_values_smoothed = np.load(os.path.join(path2, 'v_values_smoothed.npy'))
x_tensor = np.load(os.path.join(path2, 'x_values.npy'))
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f1200_1_p14_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
Height, Width = ppf.get_img_shape(Fol_Raw)
Dt = 1/1200

test = v_values_smoothed[:,-5:,:]
test2 = x_tensor[0,-1,:]
v_avg = np.trapz(test, test2, axis = -1)/Width
# average along the vertical axis to get the average velocity as a
# function of time
v_mean = np.mean(v_avg, axis = 1)
# smooth this velocity heavily
v_filter, DUMMY, DUMMY, DUMMY = smo(v_mean, s = 100, isrobust = True)
fig, ax = plt.subplots()
ax.plot(v_filter)
# ax.plot(v_mean)
# calculate the acceleration
acc = np.gradient(v_filter, Dt)
acc_norm = np.gradient(v_mean, Dt)
acc_smo, DUMMY, DUMMY, DUMMY = smo(acc, s = 200, isrobust = True) 
fig, ax = plt.subplots()
ax.plot(acc)
ax.plot(acc_smo)
# ax.plot(acc_norm)
# the maximum acceleration is in the beginning