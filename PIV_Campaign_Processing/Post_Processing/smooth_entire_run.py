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

ppf.set_plot_parameters(20, 15, 10)
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f1200_1_p12_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
NX = ppf.get_column_amount(Fol_Sol)
NY_max = ppf.get_max_row(Fol_Sol, NX)
x, y = ppf.create_grid(NX, NY)
Height, Width = ppf.get_img_shape(Fol_Raw)

Frame0 = 362
N_T = 3000-Frame0-1
# N_T = 1187-300
N_T = 3000

profiles_u = np.zeros((N_T, NY_max, NX+2))
profiles_v = np.zeros((N_T, NY_max, NX+2))

import time
start = time.time()
# extract the padded profiles
for i in range(0, N_T):
    Load_Index = Frame0 + i
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_Sol, Load_Index, NX)
    x, y, u, v = ppf.pad(x, y, u, v, Width)
    if u.shape != NX:
        # ppf.fill_
    profiles_v[i,:,:] = v
    profiles_u[i,:,:] = u
print(time.time()-start)
    

fil = sci.firwin(N_T//20, 1 , window='hamming', fs = 100)

ID_X = 1

# # filter the data with filtfilt
# data_filt_1 = sci.filtfilt(b = fil, a = [1], x = prof_1[:, :], axis = 0, padlen = N_T//3-1, padtype = 'constant')
# data_filt_2 = sci.filtfilt(b = fil, a = [1], x = prof_2[:, :], axis = 0, padlen = N_T//3-1, padtype = 'constant')
# data_filt_3 = sci.filtfilt(b = fil, a = [1], x = prof_3[:, :], axis = 0, padlen = N_T//3-1, padtype = 'constant')
# data_filt_4 = sci.filtfilt(b = fil, a = [1], x = prof_4[:, :], axis = 0, padlen = N_T//3-1, padtype = 'constant')
# data_filt_5 = sci.filtfilt(b = fil, a = [1], x = prof_5[:, :], axis = 0, padlen = N_T//3-1, padtype = 'constant')

# q_1 = np.zeros((data_filt_1.shape[0]))
# q_2 = np.zeros((data_filt_2.shape[0]))
# q_3 = np.zeros((data_filt_3.shape[0]))
# q_4 = np.zeros((data_filt_4.shape[0]))
# q_5 = np.zeros((data_filt_5.shape[0]))

# for i in range(0, N_T):
#     q_1[i] = ppf.calc_flux(x[0,:], data_filt_1[i,:])
#     q_2[i] = ppf.calc_flux(x[0,:], data_filt_2[i,:])
#     q_3[i] = ppf.calc_flux(x[0,:], data_filt_3[i,:])
#     q_4[i] = ppf.calc_flux(x[0,:], data_filt_4[i,:])
#     q_5[i] = ppf.calc_flux(x[0,:], data_filt_5[i,:])

# fig, ax = plt.subplots(figsize = (9,5))
# ax.plot(q_1, label = 'Lowest')
# ax.plot(q_2, label = '2nd Lowest')
# ax.plot(q_3, label = '3rd Lowest')
# ax.plot(q_4, label = '4th Lowest')
# ax.plot(q_5, label = '5th Lowest')
# ax.legend(loc='upper right')
# ax.set_ylim(-1000,3250)
# ax.set_xlim(0,N_T)
# ax.grid(b = True)
# fig.tight_layout(pad = 15)
# ax.set_xlabel('Timestep')
# ax.set_ylabel('Flux [px$^2$/Frame]')
# fig.savefig('Flux_comparison_glob.png', dpi=400)
    
# h_1 = np.zeros((data_filt_1.shape[0]+1))
# h_2 = np.zeros((data_filt_2.shape[0]+1))
# h_3 = np.zeros((data_filt_3.shape[0]+1))
# h_4 = np.zeros((data_filt_4.shape[0]+1))
# h_5 = np.zeros((data_filt_5.shape[0]+1))
# h_unsmoothed = np.zeros((data_filt_1.shape[0]+1))

# for i in range(0, N_T):
#     pos = h_1[i] + ppf.calc_flux(x[0,:], data_filt_1[i,:]) / Width
#     h_1[i+1] = pos
#     pos = h_2[i] + ppf.calc_flux(x[0,:], data_filt_2[i,:]) / Width
#     h_2[i+1] = pos
#     pos = h_3[i] + ppf.calc_flux(x[0,:], data_filt_3[i,:]) / Width
#     h_3[i+1] = pos
#     pos = h_4[i] + ppf.calc_flux(x[0,:], data_filt_4[i,:]) / Width
#     h_4[i+1] = pos
#     pos = h_5[i] + ppf.calc_flux(x[0,:], data_filt_5[i,:]) / Width
#     h_5[i+1] = pos
#     pos = h_unsmoothed[i] + ppf.calc_flux(x[0,:], prof_1[i,:]) / Width
#     h_unsmoothed[i+1] = pos

# fig, ax = plt.subplots()
# ax.plot(h_1, label = 'Lowest')
# # ax.plot(h_unsmoothed, label = 'Unsmoothed')
# # ax.plot(h_1-h_2, c = 'r')
# # ax.plot(h_2-h_3, c = 'b')
# # ax.plot(h_3-h_4, c = 'y')
# # ax.plot(h_4-h_5, c = 'k')
# ax.plot(h_2, label = '2nd Lowest')
# ax.plot(h_3, label = '3rd Lowest')
# ax.plot(h_4, label = '4th Lowest')
# ax.plot(h_5, label = '5th Lowest')
# ax.legend(loc='upper right')
# # ax.set_ylim(-50,50)
# ax.set_xlim(0,N_T)
# ax.grid(b = True)
# ax.set_xlabel('Timestep')
# ax.set_ylabel('$h$[px]')
# # fig.savefig('h_prediction.png', dpi = 400)
# # fig, ax = plt.subplots()
# # ax.plot(prof_1[:, ID_X], c = 'k')
# # ax.plot(data_filt_1[:, ID_X], c = 'r')
# # ax.grid(b = True)
# # ax.set_xlim (0, N_T)
# # ax.set_ylim(-5, 10)
# # fig.savefig('Profile_near_wall.png', dpi = 400)

# fig, ax = plt.subplots()
# Ts = 10
# ax.plot(x[Ts-1], prof_1[Ts-1,:], c = 'g', label = 'Unfiltered')
# ax.plot(x[Ts-1], data_filt_1[Ts-1,:], c = 'r', label = 'Filtered')
# ax.grid(b = True)
# ax.legend()
# ax.set_xlim(0, Width)
# ax.set_ylim(0, 15)
# fig.savefig('Filtering, no padding.png', dpi = 100)
# import imageio
# IMAGES_ROI = []
# IMAGES_PROF = []
# Gif_Name = 'New_profiles.gif'
# Fol_Img = ppf.create_folder('Fol_Img') 
# STP = 10
# for i in range(0,270):
#     print('Image %d of %d' %((i+1,N_T)))
#     LOAD_IDX = i*STP
#     fig, ax = plt.subplots()
#     ax.plot(x[0,:], data_filt_1[LOAD_IDX,:], label = 'Lowest')
#     ax.plot(x[0,:], data_filt_3[LOAD_IDX,:], label = '3rd Lowest')
#     ax.plot(x[0,:], data_filt_5[LOAD_IDX,:], label = '5th Lowest')
#     ax.legend()
#     ax.set_ylim(-7,14)
#     ax.set_xlim(0,x[0,-1])
#     ax.grid(b = True)
#     Name_Out = Fol_Img + os.sep + 'profile_%06d.png' %(LOAD_IDX+Frame0)
#     fig.savefig(Name_Out, dpi = 55)
#     IMAGES_PROF.append(imageio.imread(Name_Out))
#     plt.close(fig)

#     name = Fol_Raw + os.sep + 'R_h1_f1200_1_p12' +'.%06d.tif' %(LOAD_IDX+1) # because we want frame_b
#     img = cv2.imread(name,0)
#     # create the figure
#     fig, ax = plt.subplots(figsize = (2.5,8))
#     ax.imshow(img, cmap=plt.cm.gray) # show the image
#     # ax.set_yticks(np.arange(img.shape[0]-20*SCALE-1,img.shape[0],4*SCALE)) # set custom y ticks
#     # ax.set_yticklabels(np.linspace(0,20,6,dtype=int)[::-1], fontsize=15) # set custom y ticklabels
#     # ax.set_xticks(np.linspace(0,IMG_WIDTH-1, 6)) # set custom x ticks 
#     # ax.set_xticklabels(np.arange(0,6,1), fontsize=15) # set custom x ticklabels
#     ax.set_xlabel('$x$[mm]', fontsize=20) # set x label
#     ax.set_ylabel('$y$[mm]', fontsize=20) # set y label
#     fig.tight_layout(pad=1.1) # crop edges of the figure to save space
#     # plot a horizontal line of the predicted interface in case it is visible
#     if (h_5[LOAD_IDX-Frame0]+450) < Height:
#         interface_line = np.ones((img.shape[1],1))*-(h_5[LOAD_IDX-Frame0]-Height+450)
#         ax.plot(interface_line, lw = 1, c='r')
#     Name_Out = Fol_Img+os.sep+'roi_%06d.png'%LOAD_IDX # set the output name
#     fig.savefig(Name_Out, dpi = 50) # save the figure
#     plt.close(fig) # close to not overcrowd
#     IMAGES_ROI.append(imageio.imread(Name_Out)) # append into list
# imageio.mimsave(Gif_Name, IMAGES_ROI, duration = 0.05) 
# imageio.mimsave(Gif_Name, IMAGES_PROF, duration = 0.05)      

   