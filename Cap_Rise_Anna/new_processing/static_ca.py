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
Threshold_Outlier = 0.20
# threshold for the kernel filtering
Threshold_Kernel= 0.02
# pixels to cut near the wall
Wall_Cut = 3 
# whether to mirror the right side onto the left
Do_Mirror = True

"""locate the images"""
# letter of the current run
Test_Case = 'middle_A'
Case = 'Fall'
Fluid = 'Water'

Fol_Data, Pressure, Run, H_Final, Frame0, Crop_Index, Speed =\
    imp.get_parameters(Test_Case, Case, Fluid)
N_T = Frame0+Img_Amount-1
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 
# name of the output folder
Fol_Out = ppf.create_folder('Static_Ca')

"""initialize arrays"""
h_mm_avg = np.zeros([N_T+1])*np.nan  
h_cl_left = np.zeros([N_T+1])*np.nan  
h_cl_right = np.zeros([N_T+1])*np.nan  
angle_left = np.zeros([N_T+1])*np.nan  
angle_right = np.zeros([N_T+1])*np.nan 
# fit_coordinates = np.zeros((Img_Amount))


"""gif setup"""
images = [] # empty list to append into
GIFNAME = 'Detected_interface.gif' # name of the gif

# loop over all the images
for i in range(0,Img_Amount):
    # get the index starting from N_T
    Idx = i
    # get the index starting from 0
    Load_Idx = Frame0+Idx+0
    MEX= 'Exporting Im '+ str(i+1)+' of ' + str(Img_Amount) # update on progress
    print(MEX) 
    # load the image and highpass filter it
    img_hp, img = imp.load_image(Fol_Data, Crop_Index, Idx, Load_Idx, Pressure, Run, Speed)
    # calculate the detected interface position
    grad_img,y_index, x_index = imp.edge_detection_grad(img_hp,\
           Threshold_Gradient, Wall_Cut, Threshold_Outlier, Threshold_Kernel,
           do_mirror = Do_Mirror)
    # fit a gaussian to the detected interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(\
        grad_img ,Pix2mm, l=5, sigma_f=0.1, sigma_y=0.5e-6)
        
    X_c = X - 2.5 + 0.5*Pix2mm # shift the x values to be centered around 0
    subt = np.min(mu_s) # min subtraction of the y-values
    Y_c = mu_s - subt # subtract the minimum
    Y_fit = imp.fitting_cosh(X_c, Y_c) # fit to a cosine
    Y_c += subt # add the min again
    Y_fit += subt # add the min again
    
    # plot to compare the two fits
    fig, ax = plt.subplots()
    ax.plot(X_c, Y_c, label = 'Raw')
    ax.plot(X_c, Y_fit, label = 'Exponential Fit')
    ax.legend(loc = 'upper center', ncol = 2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(41.3, 43)
    ax.set_xlabel('$x$[mm]')
    ax.set_ylabel('$y$[mm]')
    ax.set_xticks(np.linspace(-2.5, 2.5, 11))
    fig.tight_layout()
    Name_Out = Fol_Out + os.sep + Test_Case + '_exponential_Fit.png'
    fig.savefig(Name_Out, dpi = 400)
    plt.close(fig)
        
    # save the values into the arrays
    h_mm_avg[Idx] = imp.vol_average(mu_s,X,img_width_mm)
    h_cl_left[Idx] = mu_s[0]
    h_cl_right[Idx] = mu_s[-1]
    angle_left[Idx]= imp.contact_angle(mu_s,X,0)
    angle_right[Idx]= imp.contact_angle(mu_s,X,-1)
    
    # plot the result with the interface detection
    fig, ax = plt.subplots() # create a figure
    mu_s = mu_s/Pix2mm # convert back to pixels
    ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    ax.scatter(i_x, i_y, marker='x', s=(13./fig.dpi)**2) # plot the detected gradient onto the image
    ax.plot((X)/(Pix2mm), mu_s, 'r-', linewidth=0.5) # plot the interface fit
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off') # disable the showing of the axis
    ax.set_ylim(mu_s[500]-20, mu_s[500]+65)
    NAME_OUT = Fol_Out + os.sep + Test_Case + '_static_ca_fit.png'
    fig.tight_layout()
    fig.savefig(NAME_OUT, dpi= 400) # save image
    plt.close(fig)
    
    # plot the result without the interface detection
    fig, ax = plt.subplots() # create a figure
    ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off') # disable the showing of the axis
    ax.set_ylim(mu_s[500]-20, mu_s[500]+65)
    NAME_OUT = Fol_Out + os.sep + Test_Case + '_static_ca_no_fit.png'
    fig.tight_layout()
    fig.savefig(NAME_OUT, dpi= 400) # save image
    plt.close(fig)