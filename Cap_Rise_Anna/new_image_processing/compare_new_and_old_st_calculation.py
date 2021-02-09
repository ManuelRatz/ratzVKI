# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:35:47 2020
@author: ratz
@description: testing of different settings for the imageprocessing to get better
edges for the detection of the contact angle .
This is a modification of Anna's codes.
"""

import numpy as np                              # for array calculations
import cv2                                      # for image processing
from matplotlib import pyplot as plt            # for plotting
import os                                       # for filepaths
import Image_processing_functions as imgprocess # for interface detection

FPS = 500 # camera frame rate in frame/sec
# tests = ['test1','test2','test3']
N_START = 1121# first image where the interface appears (check the images)
N_T = N_START+2000
THRESHOLD = 0.7  # threshold for the gradient 
OUTL_THR = 0.3 # threshold for outliers
WALL_CUT = 2 # Cutting pixels near the wall
# crop_index = (66, 210, 0, 1280) # Pixels to keep (xmin, xmax, ymin, ymax)
crop_index = (68,210,0,1280)
WIDTH = 5 # width of the channel
PIX2MM = WIDTH/(crop_index[1]-crop_index[0]) # pixel to mm

# input of the name of the current run
CURR_RUN = 'A'
TESTNAME = '1500' + CURR_RUN + '_' # prefix of the images
Fol_In = 'C:\Pa1500' + os.sep + 'Run_'+CURR_RUN + os.sep + 'images'
NAME = Fol_In + os.sep + TESTNAME # file name
# create the output folder
Fol_Out= 'C:\Pa1500'+os.sep+ 'Run_'+CURR_RUN + os.sep + 'images_detected' + os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

# dummys to fill for the loop, they are larger than necessary to synchronize with the labview data
h_mm_adv = np.zeros([N_T+1])*np.nan  
h_cl_left_all_adv = np.zeros([N_T+1])*np.nan  
h_cl_right_all_adv = np.zeros([N_T+1])*np.nan  
angle_all_left_adv = np.zeros([N_T+1])*np.nan  
angle_all_right_adv = np.zeros([N_T+1])*np.nan  

sur_t_old = np.zeros([N_T+1])*np.nan  
sur_t_new = np.zeros([N_T+1])*np.nan  

IMG_AMOUNT = N_T # amount of images to process


# function to get the surface tension by integrating the curvature
def integrate_curvature(y, x):
    # calculate the first derivative
    dydx = np.gradient(y[:,0],x)
    # calculate the second derivative
    ddyddx = np.gradient(dydx, x)
    # calculate the curvature
    curvature = ddyddx/(1+dydx**2)**1.5
    # pad the edges
    curvature[0] = curvature[2]
    curvature[1] = curvature[2]
    curvature[-1] = curvature[-3]
    curvature[-2] = curvature[-3]
    # calculate the surface tension term and return
    surface_force2 = np.trapz(curvature, x)
    return surface_force2
    
STP = 1 # use every STPth image to reduce calculation time
# iterate over all images 
for k in range(0,2000//STP+1):
    idx = N_START+STP*k # get the first index
    image = NAME + '%05d' %idx + '.png'  # file name
    img=cv2.imread(image,0)  # read the image
    img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]  # crop
    dst = cv2.fastNlMeansDenoising(img,2,2,7,21) # denoise
    dst2 = 3*dst.astype(np.float64) # increase contrast
    
    # calculate the gradient image
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(dst2,THRESHOLD,WALL_CUT,OUTL_THR, do_mirror = True) # calculate the position of the interface
    # calculate the detected interface from the gradient image
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,PIX2MM,l=5,sigma_f=1,sigma_y=6e-6) # fit a gaussian
    
    # calculate the 5 data points and save in the arrays
    h_mm_adv[idx] = imgprocess.vol_average(mu_s[:,0],X,img_width_mm)    #mm  # differences to equilibrium height
    h_cl_left_all_adv[idx] = mu_s[0]
    h_cl_right_all_adv[idx] = mu_s[-1]
    angle_all_left_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,0)
    angle_all_right_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,-1)
    
    # save the surface tension terms in separate arrays
    sur_t_old[idx] = 2*np.cos(imgprocess.contact_angle(mu_s[:,0],X,-1)*np.pi/180)
    sur_t_new[idx] = integrate_curvature(mu_s, X)
    
    MEX= 'Processing Im '+ str(idx+1)+' of ' + str(IMG_AMOUNT) # update on progress
    print(MEX) 


# create a figure to compare the two methods directly
fig, ax = plt.subplots()
ax.scatter(sur_t_old[1121::STP], sur_t_new[1121::STP], marker='o', s=(60./fig.dpi)**2) # scatter the cropped arrays
ax.set_aspect('equal') # set equal aspect ratio
x = np.linspace(0, 2, 101) # set dummy array to plot a straigt
ax.plot(x,x, c = 'r', lw = 0.5) # plot a straigt with inclination 1 to compare
ax.grid(b=True) # enable the grid
ax.set_xlabel('Old Surface Tension') # set x label
ax.set_ylabel('New Surface Tension') # set y label
ax.set_xlim(1.5, 2) # set x limits
ax.set_ylim(1.5,2.1) # set y limits
fig.savefig('comparison_theta.png', dpi = 600) # save the figure

# import filter to smooth signals
from scipy.signal import savgol_filter
# create a figure to compare the two methods in time
fig, ax = plt.subplots()
ax.plot(savgol_filter(sur_t_old[1121::STP], 15, 1, axis = 0), c = 'r', label = 'Old calculation') # plot the old calculation
ax.plot(savgol_filter(sur_t_new[1121::STP], 15, 1, axis = 0), label = 'New calculation') # plot the new calculation
ax.legend(loc = 'lower right') # show the legend
ax.set_xlim(0,2000) # set x limits
ax.set_ylim(-0.75, 2) # set y limits
ax.set_xlabel('Frame') # set x label
ax.set_ylabel('$F_{st}$') # set y label
ax.grid(b=True) # enable the grid
fig.savefig('comparison_theta_over_time.png', dpi = 600) # save the figure
