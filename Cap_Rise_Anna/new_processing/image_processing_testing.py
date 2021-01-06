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
Img_Amount = 2000
# width of the channel
Width = 5
# acquisition frequency of the camera
Fps = 500

"""Interface Detection"""
# threshold for the gradient 
Threshold_Gradient = 5  
# threshold for outliers
Threshold_Outlier = 0.10
# threshold for the kernel filtering
Threshold_Kernel= 0.02
# pixels to cut near the wall
Wall_Cut = 3
# whether to mirror the right side onto the left
Do_Mirror = True

"""locate the images"""
# letter of the current run
Test_Case = 'P1500_C30_A'
Case = 'Rise'
Fluid = 'Water'

Fol_Data, Pressure, Run, H_Final, Frame0, Crop_Index, Speed =\
    imp.get_parameters(Test_Case, Case, Fluid)
N_T = Frame0+Img_Amount-1
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 
# name of the output folder
Fol_Out = ppf.create_folder('Crop_Check')
Fol_Images_Detected = ppf.create_folder(Fol_Data + os.sep + 'Images_Detected')

"""initialize arrays"""
h_mm_avg = np.zeros([N_T+1])*np.nan  
h_cl_left = np.zeros([N_T+1])*np.nan  
h_cl_right = np.zeros([N_T+1])*np.nan  
angle_left = np.zeros([N_T+1])*np.nan  
angle_right = np.zeros([N_T+1])*np.nan 
fit_coordinates = np.zeros((1000,Img_Amount))


"""gif setup"""
images = [] # empty list to append into
GIFNAME = 'Detected_interface.gif' # name of the gif

# loop over all the images
for i in range(0,Img_Amount):
    # get the index starting from N_T
    Idx = i
    # get the index starting from 0
    Load_Idx = Frame0+Idx
    MEX= 'Exporting Im '+ str(i+1)+' of ' + str(Img_Amount) # update on progress
    print(MEX) 
    # load the image and highpass filter it
    img_hp, img = imp.load_image(Fol_Data, Crop_Index, Idx, Load_Idx,\
                                 Pressure, Run, Speed)
    # calculate the detected interface position
    grad_img,y_index, x_index = imp.edge_detection_grad(img_hp,\
           Threshold_Gradient, Wall_Cut, Threshold_Outlier, Threshold_Kernel,
           do_mirror = Do_Mirror)
    # fit a gaussian to the detected interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(\
        grad_img ,Pix2mm, l=5, sigma_f=0.1, sigma_y=0.5e-6)
        
    # save the values into the arrays
    h_mm_avg[Load_Idx] = imp.vol_average(mu_s,X,img_width_mm)
    h_cl_left[Load_Idx] = mu_s[0]
    h_cl_right[Load_Idx] = mu_s[-1]
    angle_left[Load_Idx]= imp.contact_angle(mu_s,X,0, Pix2mm)
    angle_right[Load_Idx]= imp.contact_angle(mu_s,X,-1, Pix2mm)
    fit_coordinates[:,i] = mu_s
    
    # plot the result
    fig, ax = plt.subplots() # create a figure
    mu_s = mu_s/Pix2mm # convert back to pixels
    ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    ax.scatter(i_x, i_y, marker='x', s=(13./fig.dpi)**2) # plot the detected gradient onto the image
    ax.plot((X)/(Pix2mm), mu_s, 'r-', linewidth=0.5) # plot the interface fit
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off') # disable the showing of the axis
    ax.set_ylim(mu_s[500]-20, mu_s[500]+65)
    NAME_OUT = Fol_Images_Detected + os.sep + 'testing_%05d.png' %Idx
    fig.tight_layout()
    fig.savefig(NAME_OUT, dpi= 60) # save image
    plt.close(fig) # disable or enable depending on whether you want to see image in the plot window
    images.append(imageio.imread(NAME_OUT))
# imageio.mimsave(GIFNAME, images, duration = 0.10)

pressure, f0 = imp.load_labview_files(Fol_Data, Test_Case)

h_mm_avg = h_mm_avg[Frame0:]
h_cl_left = h_cl_left[Frame0:]
h_cl_right = h_cl_right[Frame0:]
angle_left = angle_left[Frame0:]
angle_right = angle_right[Frame0:]
pressure = pressure[Frame0-f0:Frame0-f0+Img_Amount]
t = np.linspace(0,len(h_mm_avg)/Fps,len(h_mm_avg)+1)[Frame0-f0:]

imp.saveTxt(Fol_Data, h_mm_avg, h_cl_left, h_cl_right, angle_left,\
            angle_right, pressure, fit_coordinates, Test_Case)