# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 07:37:07 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:35:47 2020
@author: ratz
@description: testing of different settings for the imageprocessing to get better
edges for the detection of the contact angle .
This is a modification of Anna's codes.
"""

import numpy as np # for array calculations
import cv2 # for image processing
from matplotlib import pyplot as plt # for plotting
import os # for filepaths
import image_processing_functions as imgprocess # for interface detection
import imageio

"""General settings"""
# first image where the interface appears (check the images)
N_Start = 1015
# crop coordinates of the image
Crop_Index = (66,210,0,1280)
# amount of images to process
Img_Amount = 100 
# total number of steps (the first N_T are ignore in the calculations)
N_T = N_Start+Img_Amount-1
# width of the channel
Width = 5
# pixel to mm 
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 

"""Interface Detection"""
# threshold for the gradient 
Threshold_Gradient = 5  
# threshold for outliers
Threshold_Outlier = 0.12
# threshold for the kernel filtering
Threshold_Kernel= 0.02
# pixels to cut near the wall
Wall_Cut = 3 
# whether to mirror the right side onto the left
Do_Mirror = True

"""locate the images"""
# letter of the current run
Curr_Run = 'B'
# location of the images
Fol_In = 'C:\Pa1500' + os.sep + 'Run_' + Curr_Run + os.sep + 'images'
# prefix of the images
Run_Name = '1500' + '_' + Curr_Run 
# name of the file
Name = Fol_In + os.sep + Run_Name
# name of the output folder
Fol_Out = 'Temp_' + Curr_Run
if not os.path.exists(Fol_Out):
    os.makedirs(Fol_Out)

"""initialize arrays"""
h_mm_avg = np.zeros([N_T+1])*np.nan  
h_cl_left = np.zeros([N_T+1])*np.nan  
h_cl_right = np.zeros([N_T+1])*np.nan  
angle_left = np.zeros([N_T+1])*np.nan  
angle_right = np.zeros([N_T+1])*np.nan 

"""gif setup"""
images = [] # empty list to append into
GIFNAME = 'Detected_interface.gif' # name of the gif

# loop over all the images
for i in range(0,Img_Amount):
    # get the index starting from N_T
    Idx = i
    # get the index starting from 0
    Load_Idx = N_Start+Idx
    MEX= 'Exporting Im '+ str(i+1)+' of ' + str(Img_Amount) # update on progress
    print(MEX) 
    
    # load the image and highpass filter it
    img_hp = imgprocess.load_image(Name, Crop_Index, Idx, Load_Idx)
    # calculate the detected interface position
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(img_hp,\
           Threshold_Gradient, Wall_Cut, Threshold_Outlier, Threshold_Kernel,
           do_mirror = Do_Mirror)
    # fit a gaussian to the detected interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = \
        imgprocess.fitting_advanced(grad_img ,Pix2mm, l=5, sigma_f=0.1,
                                    sigma_y=0.5e-6)
    # save the values into the arrays
    h_mm_avg[Idx] = imgprocess.vol_average(mu_s,X,img_width_mm)
    h_cl_left[Idx] = mu_s[0]
    h_cl_right[Idx] = mu_s[-1]
    angle_left[Idx]= imgprocess.contact_angle(mu_s,X,0)
    angle_right[Idx]= imgprocess.contact_angle(mu_s,X,-1)
    
    
    # plot the result
    # fig, ax = plt.subplots() # create a figure
    # mu_s = mu_s/Pix2mm # convert back to pixels
    # ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    # ax.scatter(i_x, i_y, marker='x', s=(13./fig.dpi)**2) # plot the detected gradient onto the image
    # ax.plot((X)/(Pix2mm), mu_s, 'r-', linewidth=0.5) # plot the interface fit
    # ax.invert_yaxis()
    # ax.set_aspect('equal')
    # ax.axis('off') # disable the showing of the axis
    # ax.set_ylim(mu_s[500]-20, mu_s[500]+65)
    # NAME_OUT = Fol + os.sep + 'testing_%d.png'%Idx
    # fig.tight_layout()
    # fig.savefig(NAME_OUT, dpi= 80) # save image

    # plt.close(fig) # disable or enable depending on whether you want to see image in the plot window
    # images.append(imageio.imread(NAME_OUT))
# imageio.mimsave(GIFNAME, images, duration = 0.10)

"""save the txts"""
Fol_Out_Adv= os.path.abspath(Fol_Out + os.sep + 'Txts_advanced_fitting')
imgprocess.saveTxt(Fol_Out_Adv, h_mm_avg, h_cl_left, h_cl_right, angle_left, angle_right)