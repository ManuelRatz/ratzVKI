# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 07:37:07 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np # for array calculations
import cv2 # for image processing
from matplotlib import pyplot as plt # for plotting
import os # for filepaths
import image_processing_functions as imp # for interface detection
import post_processing_functions as ppf
import imageio

ppf.set_plot_parameters(20, 15, 10)

"""General settings"""
# amount of images to process
Img_Amount = 1
# width of the channel
Width = 5
# acquisition frequency of the camera
Fps = 500

"""Interface Detection"""
# threshold for the gradient 
Threshold_Gradient = 5
# threshold for outliers
Threshold_Int = 7
Threshold_Outlier = 0.1
# threshold for the kernel filtering
Threshold_Kernel= 0.02
# pixels to cut near the wall
Wall_Cut = 3
# whether to mirror the right side onto the left
Do_Mirror = True
Denoise = False

"""locate the images"""
# letter of the current run
Test_Case = 'P1500_C30_B'
Case = 'Rise'
Fluid = 'Water'

Fol_Data, Pressure, Run, H_Final, Frame0, Crop_Index, Speed =\
    imp.get_parameters(Test_Case, Case, Fluid)
N_T = Frame0+Img_Amount-1
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 

Idx = 1493-Frame0



"""gif setup"""
images = [] # empty list to append into
GIFNAME = 'Detected_interface.gif' # name of the gif

def func(x_func,a,b):
    return (np.cosh(np.abs(x_func)**a/b)-1)

Denoise = True
# get the index starting from 0
Load_Idx = Idx+Frame0
# load the image and highpass filter it
img_hp, img = imp.load_image(Fol_Data, Crop_Index, Idx, Load_Idx,\
                             Pressure, Run, Speed, Denoise)
# calculate the detected interface position
grad_img,y_index, x_index = imp.edge_detection_grad(img_hp,\
       Threshold_Gradient, Wall_Cut, Threshold_Outlier, Threshold_Kernel,
       Threshold_Int, do_mirror = Do_Mirror)
# fit a gaussian to the detected interface
mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(\
    grad_img ,Pix2mm, l=5, sigma_f=0.1, sigma_y=5e-7)

i_x_fit = i_x_mm - 2.5
Subt = np.min(i_y_mm)
i_y_fit = i_y_mm-Subt
X_plot = (np.linspace(-2.5,2.5,1000))
mu_s = mu_s/Pix2mm # convert back to pixels

fit_val = imp.fitting_cosh2(i_x_fit, i_y_fit)
y_cosh_fine = (func(X_plot, fit_val[0], fit_val[1])+Subt)/Pix2mm
# plot the result
fig, ax = plt.subplots() # create a figure
ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
ax.scatter(i_x, i_y, marker='x', s=(70./fig.dpi)**2, color = 'lime') # plot the detected gradient onto the image
ax.plot((X_plot+2.5)/Pix2mm-0.5, y_cosh_fine, lw=0.5, color = 'yellow')
ax.plot((X)/(Pix2mm), mu_s, 'r-', linewidth=0.5) # plot the interface fit
ax.invert_yaxis()
ax.set_aspect('equal')
ax.axis('off') # disable the showing of the axis
ax.set_ylim(mu_s[500]-20, mu_s[500]+65)
ax.set_xlim(65,img.shape[1]+1)
NAME_OUT = 'Stp_%05d.png' %(Idx+Frame0)
fig.tight_layout()
fig.savefig(NAME_OUT, dpi= 400) # save image
