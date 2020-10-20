# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:33:22 2020

@author: ratz
@description: load the PIV images and crop to get an idea of the magnification
also create an animation of a simple flow to show the phenomena
"""

import os # for data paths
import matplotlib.pyplot as plt # for showing the images
import numpy as np # for array operations
import imageio # for the animation
import cv2 # for loading the images

# first we look at the images aquired at 100 Hz
Fol_In = 'rise_f_100' + os.sep

# we load one image and try different crop settings for it
name = Fol_In + 'Export.6d1rxnak.000049.tif'
img=cv2.imread(name,0) # load the image

# set the cropping indices
width_px = 320 # image width in pixel
width_mm = 5 # width of the channel in mm
scaling_factor = width_px / width_mm # scaling factor in px/mm
available_height = 1200/scaling_factor # theoretical height that can be observed

crop= (0,1200,349,349+width_px)
img_crop = img[crop[2]:crop[3],crop[0]:crop[1]]
plt.imshow(img_crop,cmap=plt.cm.gray)
# plt.close('all')

# now we create the animation
GIFNAME = 'test.gif'

IMG_AMOUNT = 100 # amount of images to load
# set up the loop over all the images
images = []
for i in range(0, IMG_AMOUNT):
    # load and crop the image
    name = Fol_In + 'Export.6d1rxnak.%06d.tif' %(i + 19)
    jpg_name = 'tmp' + os.sep + 'Export.6d1rxnak.%06d.png' %(i + 19)
    img = cv2.imread(name,0)
    img = img[crop[2]:crop[3],crop[0]:crop[1]]
    
    # show the image    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.axis('off') # disable the showing of the axis
    plt.imshow(img,cmap=plt.cm.gray)
    plt.savefig(jpg_name, dpi = 100)
    plt.close('all')
    
    MEX = 'Processing image %d of %d' %((i+1),IMG_AMOUNT)
    print(MEX)
    images.append(imageio.imread(jpg_name))
    
imageio.mimsave(GIFNAME, images, duration = 0.2)