# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:56:58 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import post_processing_functions as ppf
import os
import cv2
import matplotlib.pyplot as plt
import scipy.signal as sci 

#%%
"""
Here we load the images and the data by giving the input folders
"""

Fol_Raw = 'C:\PIV_Processed\Images_Preprocessed\F_h2_f1000_1_q'+ os.sep
Fol_In = 'C:\PIV_Processed\Images_Processed\Results_F_h2_f1000_1_q_peak2RMS_24_24' 

# give the loading index and load the image
Index = 326
image_name = Fol_Raw + 'F_h2_f1000_1_q.%06d.tif' %Index
img = cv2.imread(image_name,0)
img = np.flip(img, axis = 0)

# load the data
NX = ppf.get_column_amount(Fol_In) # amount of columns
x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, Index, NX)

#%%
"""
Here we start with the filtering of the velocity fields to look for vorticies
"""

# # Old code as examplatory testing
# n=200 # amont of samples
# fs = 100 # sampling frequency
# t = np.arange(0, n/fs, 1/fs) # set up the timesteps
# data = np.cos(t*np.pi) # create data
# data = data + np.random.random(n) * 0.2 # make data noisy

# # define the observation windows in the time domain
# fil = sci.firwin(20, 100/fs*2 , window='hamming', fs = 50)

# # filter the data with filtfilt
# data_filt = sci.filtfilt(b = fil, a = [1], x = data)

# fig, ax = plt.subplots(figsize = (8,5))
# ax.plot(t, data, label = 'Unfiltered', c = 'b')
# ax.plot(t, data_filt, label = 'filtered', c = 'r')
# ax.set_xlabel('t[s]')
# ax.set_ylabel('cos(t)')
# ax.set_xlim([0, n//fs])
# ax.set_ylim(-1, 1.2)


#%%
"""
This is cosmetics to make the plot look good and finally plot the result
"""

# give the steps in case we want to skip rows and columns
Stpx = 1
Stpy = 1
x = x[::Stpy, ::Stpx]
y = y[::Stpy, ::Stpx]
u = u[::Stpy, ::Stpx]
v = v[::Stpy, ::Stpx]
ratio = ratio[::Stpy, ::Stpx]
mask = mask[::Stpy, ::Stpx]
invalid = mask.astype('bool')
valid = ~invalid # mask for the colors

# create the figure
fig, ax = plt.subplots()
ax.imshow(img, cmap = plt.cm.gray) # show the iamge
ax.axis('off') # disable the axis
ax.invert_yaxis() # because image is inverted
# scaling parameters for the plots
Scale = 400 # for the arrow length
Width = 0.00225 # for the width of the base
Headwidth = 4.5 # for the width of the head
Headaxislength = 2.5 # for the length of the head
# quiver plots, valid and invalid
ax.quiver(x[valid], y[valid], u[valid], v[valid], color = 'g', scale = Scale,\
          width = Width, headwidth = Headwidth, headaxislength = Headaxislength)
ax.quiver(x[invalid], y[invalid], u[invalid], v[invalid], color = 'r', scale = Scale,\
          width = Width, headwidth = Headwidth, headaxislength = Headaxislength)
ax.set_ylim(700,1100) # to better see the part close to the surface
fig.savefig('Vorticies_fall.png', dpi = 600) # save the result