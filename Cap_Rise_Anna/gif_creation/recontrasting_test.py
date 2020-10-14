#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:26:07 2020

@author: ratz
@description: Testing of recontrasting
"""


from skimage.io import imread, imshow, imsave # for reading and showing
import cv2 # for reading images
import matplotlib.pyplot as plt # for plotting
import numpy as np # for arrays
import os # for datah paths
import Image_processing_functions as imgprocess # for edge detection

#load the image
Name_img = 'test_images' + os.sep + '350Pa_C01277.png'
Im = cv2.imread(Name_img, 0)


wall_cut = 0 # Cutting pixels near the wall
treshold_pos = 1  # threshold for the gradient 
crop_indx1 = 66  # first x coordinate to crop the image
crop_indx2 = 210 # second x coordinate to crop the image
crop_indy1 = 0 # first y coordinate to crop the image
crop_indy2 = 1280 # second y coordinate to crop the image
width = 5 # width of the channel
pix2mm = width/(crop_indx2-crop_indx1) # pixel to mm

# crop the image
Im_denoise = Im[crop_indy1:crop_indy2,crop_indx1:crop_indx2]
# remove background noise
# Im_denoise = cv2.fastNlMeansDenoising(Im_denoise,10,1,7,21)

# get the histogramm and show it
equ = cv2.equalizeHist(Im_denoise)
res = np.hstack((Im_denoise, equ)) #stacking images side-by-side
imshow(res)

# get the histogram and show it
hist = cv2.calcHist([Im_denoise],[0],None,[256],[0,256])
plt.figure()
plt.plot(hist)

"""
this is the part imported from Anna's codes
"""

# detect the interface and plot it
grad_img,y_index, x_index = imgprocess.edge_detection_grad(equ,treshold_pos,wall_cut)

mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,pix2mm,l=5,sigma_f=2000,sigma_y=10)

mu_s = mu_s/pix2mm
crop_img1 = equ[int(1280-mu_s[500])-50:int(1280-mu_s[500])+50,0:144]
    
plt.figure()
plt.imshow(crop_img1, cmap=plt.cm.gray)
plt.plot((X)/(pix2mm)-0.5, -mu_s+mu_s[500]+50, 'r-', linewidth=0.5)
#plt.plot(i_x,len(grad_img[:,0])-i_y,'x')
plt.axis('off')
Name= 'images_detected' + os.sep +'Step_'+'.png'
# MEX= 'Exporting Im '+ str(1)+' of ' + '2000'
# print(MEX)
# plt.grid()
# plt.title('Image %04d' % (idx-n_start+1))
plt.savefig(Name)
                                             