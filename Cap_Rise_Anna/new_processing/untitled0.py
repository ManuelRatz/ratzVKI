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
import Image_processing_functions as imgprocess # for interface detection
import imageio

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

CURR_RUN = 'A'

TESTNAME = '1500' + CURR_RUN + '_' # prefix of the images
Fol_In = 'C:\Pa1500' + os.sep + 'Run_'+CURR_RUN + os.sep + 'images'
NAME = Fol_In + os.sep + TESTNAME # file name
# create the output folder
Fol_Out= 'C:\Pa1500'+os.sep+ 'Run_'+CURR_RUN + os.sep + 'images_detected' + os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

h_mm_adv = np.zeros([N_T+1])*np.nan  
h_cl_left_all_adv = np.zeros([N_T+1])*np.nan  
h_cl_right_all_adv = np.zeros([N_T+1])*np.nan  
angle_all_left_adv = np.zeros([N_T+1])*np.nan  
angle_all_right_adv = np.zeros([N_T+1])*np.nan  

sur_t_old = np.zeros([N_T+1])*np.nan  
sur_t_new = np.zeros([N_T+1])*np.nan  

IMG_AMOUNT = N_T # amount of images to process
plus = 0

images = []
GIFNAME = 'Detected interface'

def integrate_curvature(y, x):
    # x = np.expand_dims(x, axis = 1)
    grad = np.gradient(y[:,0],x)
    gradgrad = np.gradient(grad, x)
    curvature = gradgrad/(1+grad**2)**1.5
    curvature[0] = curvature[2]
    curvature[1] = curvature[2]
    curvature[-1] = curvature[-3]
    curvature[-2] = curvature[-3]
    surface_force2 = np.trapz(curvature, x)
    return surface_force2
# iterate over all images 
    
STP = 1
for k in range(0,2000//STP+1):
    idx = N_START+STP*k+plus # get the first index
    image = NAME + '%05d' %idx + '.png'  # file name
    img=cv2.imread(image,0)  # read the image
    img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]  # crop
    dst = cv2.fastNlMeansDenoising(img,2,2,7,21) # denoise
    dst2 = 3*dst.astype(np.float64) # increase contrast
 
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(dst2,THRESHOLD,WALL_CUT,OUTL_THR, do_mirror = True) # calculate the position of the interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,PIX2MM,l=5,sigma_f=1,sigma_y=6e-6) # fit a gaussian

    h_mm_adv[idx] = imgprocess.vol_average(mu_s[:,0],X,img_width_mm)    #mm  # differences to equilibrium height
    h_cl_left_all_adv[idx] = mu_s[0]
    h_cl_right_all_adv[idx] = mu_s[-1]
    angle_all_left_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,0)
    angle_all_right_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,-1)
    
    sur_t_old[idx] = 2*np.cos(imgprocess.contact_angle(mu_s[:,0],X,-1)*np.pi/180)
    sur_t_new[idx] = integrate_curvature(mu_s, X)
    
    mu_s = mu_s/PIX2MM # calculate the resulting height in mm
    # plot the result
    # final_img = img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+50,0:500]
    # grad_img = grad_img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+50,0:500]
    # fig, ax = plt.subplots() # create a figure
    # plt.imshow(final_img, cmap=plt.cm.gray) # show the image in greyscale
    # plt.scatter(i_x, -i_y+mu_s[500]+70, marker='x', s=(13./fig.dpi)**2) # plot the detected gradient onto the image
    # plt.plot((X)/(PIX2MM)-0.5, -mu_s+mu_s[500]+70, 'r-', linewidth=0.5) # plot the interface fit
    # plt.axis('off') # disable the showing of the axis
    # NAME_OUT=Fol_Out+ os.sep +'Step_'+str(idx)+'.png' # set output name
    MEX= 'Exporting Im '+ str(idx+1)+' of ' + str(IMG_AMOUNT) # update on progress
    print(MEX) 
    # plt.title('Image %04d' % ((idx-1121))) # set image title
    # plt.close(fig) # disable or enable depending on whether you want to see image in the plot window
    # images.append(imageio.imread(NAME_OUT))

print('Done')
fig, ax = plt.subplots()
ax.scatter(sur_t_old[1121::STP], sur_t_new[1121::STP], marker='o', s=(60./fig.dpi)**2)
# ax.plot(sur_t_old[1121::STP], c = 'r')
# ax.plot(sur_t_new[1121::STP])
x = np.linspace(0, 2, 101)
ax.set_aspect('equal')
ax.plot(x,x, c = 'r', lw = 0.5)
ax.grid(b=True)
ax.set_xlabel('Old Surface Tension')
ax.set_ylabel('New Surface Tension')
ax.set_xlim(1.5, 2)
ax.set_ylim(1.5,2.1)
fig.savefig('comparison_theta.png', dpi = 600)

from scipy.signal import savgol_filter
fig, ax = plt.subplots()
ax.plot(savgol_filter(sur_t_old[1121::STP], 15, 1, axis = 0), c = 'r', label = 'Old calculation')
ax.plot(savgol_filter(sur_t_new[1121::STP], 15, 1, axis = 0), label = 'New calculation')
ax.legend(loc = 'lower right')
ax.set_xlim(0,2000)
ax.set_ylim(-0.75, 2)
ax.grid(b=True)
fig.savefig('comparison_theta_over_time.png', dpi = 600)
