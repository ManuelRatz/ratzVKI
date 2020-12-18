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
from scipy.ndimage import gaussian_filter

FPS = 500 # camera frame rate in frame/sec
# tests = ['test1','test2','test3']
N_START = 1121# first image where the interface appears (check the images)
N_T = N_START+2000
THRESHOLD = 12  # threshold for the gradient 
OUTL_THR = 0.3 # threshold for outliers
WALL_CUT = 2 # Cutting pixels near the wall
# crop_index = (66, 210, 0, 1280) # Pixels to keep (xmin, xmax, ymin, ymax)
crop_index = (66,210,0,1280)
WIDTH = 5 # width of the channel
PIX2MM = WIDTH/(crop_index[1]-crop_index[0]) # pixel to mm

TESTNAME = '1500_A05000.png' # prefix of the images
Fol_In = 'G:\Anna\WATER\\20200825\\1500Pa\\test1\images'
NAME = Fol_In + os.sep + TESTNAME # file name
# create the output folder


# h_mm_adv = np.zeros([N_T+1])*np.nan  
# h_cl_left_all_adv = np.zeros([N_T+1])*np.nan  
# h_cl_right_all_adv = np.zeros([N_T+1])*np.nan  
# angle_all_left_adv = np.zeros([N_T+1])*np.nan  
# angle_all_right_adv = np.zeros([N_T+1])*np.nan  

IMG_AMOUNT = N_T # amount of images to process
plus = 0

# Images = []
GIFNAME = 'Detected interface'
# iterate over all images 

image = NAME  # file name
img=cv2.imread(image,0)  # read the image
img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]  # crop
# blur = gaussian_filter(img, sigma = 11, mode = 'nearest', truncate = 3)
# dst = img - blur
# dst[dst<0] = 0
# dst = dst/dst.max()*255
# dst = dst.astype(np.uint8)
dst = cv2.fastNlMeansDenoising(img,2,2,7,21) # denoise
# dst = dst-np.min(dst) 
# dst[dst<0] = 0

grad_img,y_index, x_index = imgprocess.edge_detection_grad(dst,THRESHOLD,WALL_CUT,OUTL_THR, do_mirror = True) # calculate the position of the interface
mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,PIX2MM,l=5,sigma_f=0.1,sigma_y=0.5e-6) # fit a gaussian

# grad_img2,y_index2, x_index2 = imgprocess.edge_detection_grad(dst2,THRESHOLD,WALL_CUT,OUTL_THR, do_mirror = False) # calculate the position of the interface
# mu_s2,i_x,i_y,i_x_mm,i_y_mm,X2,img_width_mm = imgprocess.fitting_advanced(grad_img2,PIX2MM,l=5,sigma_f=1,sigma_y=6e-6) # fit a gaussian

# h_mm_adv[idx] = imgprocess.vol_average(mu_s[:,0],X,img_width_mm)    #mm  # differences to equilibrium height
# h_cl_left_all_adv[idx] = mu_s[0]
# h_cl_right_all_adv[idx] = mu_s[-1]
# angle_all_left_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,0)
# angle_all_right_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,-1)

mu_s = mu_s/PIX2MM # calculate the resulting height in mm
# mu_s = mu_s[:-6]
# X = X[:-6]

# plot the result
# dst = dst-np.min(dst)
final_img = img[int(img.shape[0]-mu_s[500])-70:int(img.shape[0]-mu_s[500])+50,0:500]
grad_img = grad_img[int(img.shape[0]-mu_s[500])-70:int(img.shape[0]-mu_s[500])+50,0:500]
fig, ax = plt.subplots() # create a figure
ax.imshow(final_img, cmap=plt.cm.gray) # show the image in greyscale
ax.scatter(i_x, -i_y+mu_s[500]+70, marker='x', s=(45./fig.dpi)**2, color = 'red') # plot the detected gradient onto the image
ax.plot((X)/(PIX2MM), -mu_s+mu_s[500]+70, linewidth=0.5, color  = 'lime') # plot the interface fit
ax.axis('off') # disable the showing of the axis
NAME_OUT='static_ca.png' # set output name
fig.tight_layout()
fig.savefig(NAME_OUT, dpi= 500) # save image
