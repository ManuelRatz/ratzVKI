# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:52:58 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

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
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Fall_24_24_peak2RMS\Results_F_h3_f1200_1_s_24_24'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
NX = ppf.get_column_amount(Fol_Sol)
NY_max = ppf.get_max_row(Fol_Sol, NX)
# x, y = ppf.create_grid(NX, NY)
Height, Width = ppf.get_img_shape(Fol_Raw)

Frame0 = 136
N_T = 486-Frame0-1
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

