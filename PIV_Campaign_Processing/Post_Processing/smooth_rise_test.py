# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:59:46 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import post_processing_functions as ppf
import matplotlib.pyplot as plt
import scipy.signal as sci 
import os
import cv2
from smoothn import smoothn

ppf.set_plot_parameters(20, 15, 10)
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f1200_1_p12_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
NX = ppf.get_column_amount(Fol_Sol)
NY_max = ppf.get_max_row(Fol_Sol, NX)
# x, y = ppf.create_grid(NX, NY)
Height, Width = ppf.get_img_shape(Fol_Raw)

Frame0 = 361
N_T = 3000-Frame0-1
# N_T = 1187-300

profiles_u = np.zeros((N_T, NY_max, NX+2))
profiles_v = np.zeros((N_T, NY_max, NX+2))

import time
start = time.time()
# extract the padded profiles
for i in range(0, N_T):
    Load_Index = Frame0 + i
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_Sol, Load_Index, NX)
    x, y, u, v = ppf.pad(x, y, u, v, Width)
    u, v = ppf.fill_zeros(u, v, NY_max)
    profiles_v[i,:,:] = v
    profiles_u[i,:,:] = u
print(time.time()-start)
    
column = 1
row_clip = 4
row_cont = 30


fil = sci.firwin(N_T//10, 0.5 , window='hamming', fs = 100)

fig, ax = plt.subplots()
ax.plot(profiles_v[:,row_clip,column], lw = 0.3, label = 'Unsmoothed')
ax.set_xlim(0, N_T)
ax.set_ylim(-4,6)
data_smoothn, Dum, Dum, Dum = smoothn(profiles_v[:,:,column], axis = 0, s = 500)
ax.plot(data_smoothn[:,row_clip], lw = 0.6, label = 'Smoothn', c = 'lime')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_v, axis = 0, padlen = N_T//3-1, padtype = 'constant')
ax.plot(data_filt[:,row_clip,column], lw = 0.6, label = 'Filtfilt', c = 'r')
ax.legend()
Name = 'interrupted_signal_v.png'
fig.savefig(Name, dpi = 400)

fil2 = sci.firwin(N_T//20, 0.5 , window='hamming', fs = 100)
fig, ax = plt.subplots()
ax.plot(profiles_u[:,row_clip,column], lw = 0.3, label = 'Unsmoothed')
ax.set_xlim(0, N_T)
ax.set_ylim(-2,2)
data_smoothn, Dum, Dum, Dum = smoothn(profiles_u[:,:,column], axis = 0, s = 500)
ax.plot(data_smoothn[:,column], label = 'Smoothn', lw = 0.6, c = 'lime')
data_filt = sci.filtfilt(b = fil2, a = [1], x = profiles_u, axis = 0, padlen = N_T//3-1, padtype = 'constant')
ax.plot(data_filt[:,column,column], label = 'Filtfilt', lw = 0.6, c = 'r')
ax.legend()
Name = 'interrupted_signal_u.png'
fig.savefig(Name, dpi = 400)

fig, ax = plt.subplots()
ax.plot(profiles_v[:,row_cont,column], lw = 0.3, label = 'Unsmoothed')
ax.set_xlim(0, N_T)
# ax.set_ylim(-4,6)
data_smoothn, Dum, Dum, Dum = smoothn(profiles_v[:,:,column], axis = 0, s = 500)
ax.plot(data_smoothn[:,row_cont], label = 'Smoothn', c = 'lime')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_v, axis = 0, padlen = N_T//3-1, padtype = 'constant')
ax.plot(data_filt[:,row_cont,column], label = 'Filtfilt', c = 'r')
ax.legend()
Name = 'continuous_signal_v.png'
fig.savefig(Name, dpi = 400)

fig, ax = plt.subplots()
ax.plot(profiles_u[:,row_cont,column], lw = 0.3, label = 'Unsmoothed')
ax.set_xlim(0, N_T)
# ax.set_ylim(-4,6)
data_smoothn, Dum, Dum, Dum = smoothn(profiles_u[:,:,column], axis = 0, s = 500)
ax.plot(data_smoothn[:,row_cont], label = 'Smoothn', c = 'lime')
data_filt = sci.filtfilt(b = fil2, a = [1], x = profiles_u, axis = 0, padlen = N_T//3-1, padtype = 'constant')
ax.plot(data_filt[:,row_cont,column], label = 'Filtfilt', c = 'r')
ax.legend()
Name = 'continuous_signal_u.png'
fig.savefig(Name, dpi = 400)

time_step = 1
fil = sci.firwin(8, 0.005 , window='hamming', fs = 100)
fig, ax = plt.subplots()
ax.set_title('Timestep %d, Column %d' %(time_step, column))
ax.plot(profiles_v[time_step,:,column], label = 'Unsmoothed')
data_smoothn, Dum, Dum, Dum = smoothn(profiles_v[time_step,:,column], axis = 0, s =5)
ax.plot(data_smoothn, c = 'lime', label = 'Smoothn')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_v[time_step,:,column], axis = 0, padtype = 'constant')
ax.plot(data_filt, c = 'r', label = 'Filtfilt')
ax.legend(loc = 'upper right')
Name = 'left_column_v'
fig.savefig(Name, dpi = 400)

fil = sci.firwin(8, 0.005 , window='hamming', fs = 100)
fig, ax = plt.subplots()
ax.set_title('Timestep %d, Column %d' %(time_step, column))
ax.plot(profiles_u[time_step,:,column], label = 'Unsmoothed')
data_smoothn, Dum, Dum, Dum = smoothn(profiles_u[time_step,:,column], axis = 0, s =5)
ax.plot(data_smoothn, c = 'lime', label = 'Smoothn')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_u[time_step,:,column], axis = 0, padtype = 'constant')
ax.plot(data_filt, c = 'r', label = 'Filtfilt')
ax.legend(loc = 'lower right')
Name = 'left_column_u'
fig.savefig(Name, dpi = 400)

ppf.set_plot_parameters(20, 15, 10)
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f750_1_p14_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
NX = ppf.get_column_amount(Fol_Sol)
NY_max = ppf.get_max_row(Fol_Sol, NX)
# x, y = ppf.create_grid(NX, NY)
Height, Width = ppf.get_img_shape(Fol_Raw)

Frame0 = 183
N_T = 1875-Frame0-1
# N_T = 1187-300

profiles_u2 = np.zeros((N_T, NY_max, NX+2))
profiles_v2 = np.zeros((N_T, NY_max, NX+2))

import time
start = time.time()
# extract the padded profiles
for i in range(0, N_T):
    Load_Index = Frame0 + i
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_Sol, Load_Index, NX)
    x, y, u, v = ppf.pad(x, y, u, v, Width)
    u, v = ppf.fill_zeros(u, v, NY_max)
    profiles_v2[i,:,:] = v
    profiles_u2[i,:,:] = u
print(time.time()-start)
    
column = 2
row_clip = 4
row_cont = 30

time_step = 20
fil = sci.firwin(8, 0.005 , window='hamming', fs = 100)
fig, ax = plt.subplots()
ax.set_title('Timestep %d, Column %d' %(time_step, column))
ax.plot(profiles_v2[time_step,:,column], label = 'Unsmoothed')
data_smoothn, dum, dum, dum = smoothn(profiles_v2[time_step,:,column], s = 10)
ax.plot(data_smoothn, label = 'Smoothn')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_v2[time_step,:,column], axis = 0, padtype = 'constant')
ax.plot(data_filt, label = 'Filtfilt')
ax.legend(loc = 'lower right')
Name = 'left_column_u_bad'
fig.savefig(Name, dpi = 400)

fig, ax = plt.subplots()
ax.set_title('Timestep %d, Column %d' %(time_step, column))
ax.plot(profiles_u2[time_step,:,column], label = 'Unsmoothed')
data_smoothn, dum, dum, dum = smoothn(profiles_u2[time_step,:,column], s = 10)
ax.plot(data_smoothn, label = 'Smoothn')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_u2[time_step,:,column], axis = 0, padtype = 'constant')
ax.plot(data_filt, label = 'Filtfilt')
ax.legend(loc = 'lower right')
Name = 'left_column_u_bad'
fig.savefig(Name, dpi = 400)


fil = sci.firwin(N_T//20, 0.5 , window='hamming', fs = 100)

fig, ax = plt.subplots()
ax.plot(profiles_v2[:,row_clip,column], lw = 0.3, label = 'Unsmoothed')
ax.set_xlim(0, N_T)
ax.set_ylim(-7,10)
data_smoothn, Dum, Dum, Dum = smoothn(profiles_v2[:,:,column], axis = 0, s = 500)
ax.plot(data_smoothn[:,row_clip], lw = 0.6, label = 'Smoothn', c = 'lime')
data_filt = sci.filtfilt(b = fil, a = [1], x = profiles_v2, axis = 0, padlen = N_T//3-1, padtype = 'constant')
ax.plot(data_filt[:,row_clip,column], lw = 0.6, label = 'Filtfilt', c = 'r')
ax.legend()
Name = 'continuous_signal_v_bad.png'
fig.savefig(Name, dpi = 400)

fil2 = sci.firwin(N_T//20, 0.5 , window='hamming', fs = 100)
fig, ax = plt.subplots()
ax.plot(profiles_u2[:,row_clip,column], lw = 0.3, label = 'Unsmoothed')
ax.set_xlim(0, N_T)
ax.set_ylim(-0.5,0.5)
data_smoothn, Dum, Dum, Dum = smoothn(profiles_u2[:,:,column], axis = 0, s = 500)
ax.plot(data_smoothn[:,column], label = 'Smoothn', lw = 0.6, c = 'lime')
data_filt = sci.filtfilt(b = fil2, a = [1], x = profiles_u2, axis = 0, padlen = N_T//3-1, padtype = 'constant')
ax.plot(data_filt[:,column,column], label = 'Filtfilt', lw = 0.6, c = 'r')
ax.legend()
Name = 'continuous_signal_u_bad.png'
fig.savefig(Name, dpi = 400)

