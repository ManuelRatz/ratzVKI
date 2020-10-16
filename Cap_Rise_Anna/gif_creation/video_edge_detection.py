# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:35:47 2020
@author: ratz
@description: testing of different settings for the imageprocessing to get better
edges for the detection of the contact angle 

This is a modification of Anna's codes that only focuses on the processing, not
the exporting of the data'
"""

import numpy as np # for array calculations
import cv2 # for image processing
from matplotlib import pyplot as plt # for plotting
import os # for filepaths
import image_processing_functions as imgprocess # for interface detection
from skimage import exposure # for greyscaling and histogramm equalisation

FPS = 500 # camera frame rate in frame/sec
n_start = 1121# first image where the interface appears (check the images)
THRESHOLD = 0.7  # threshold for the gradient 
Outlier_threshold = 0.10 # threshold for outliers
WALL_CUT = 3 # Cutting pixels near the wall
crop_index = (66, 210, 0, 1280) # Pixels to keep (xmin, xmax, ymin, ymax)
WIDTH = 5 # width of the channel
pix2mm = WIDTH/(crop_index[1]-crop_index[0]) # pixel to mm

testname = '350Pa_C' # prefix of the images
FOL = 'images' + os.sep # input folder
name = FOL + os.sep + testname  # file name

# create the output folder
Fol_Out= 'images_detected/' + os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

IMG_AMOUNT = 200 # amount of images to process
Image = 1492
# iterate over all images 
for k in range(0,201):
    idx = n_start+10*k-1 # get the first index
    image = name + '%05d' %idx + '.png'  # file name
    img=cv2.imread(image,0)  # read the image
    img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]  # crop
    dst = cv2.fastNlMeansDenoising(img,2,2,7,21) # denoise
    dst2 = 3*dst.astype(np.float64) # increase contrast
    # for i in range(1, 20, 4):
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(dst2,THRESHOLD,WALL_CUT,Outlier_threshold) # calculate the position of the interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,pix2mm,l=5,sigma_f=0.5,sigma_y=4e-5) # fit a gaussian
    mu_s = mu_s/pix2mm # calculate the resulting height in mm
    # final_img = dst2[int(1280-mu_s[500])-60:int(1280-mu_s[500])+60,0:144] # crop the image 60 px above and 60 px below the centerpoint
    final_img = img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+70,0:144]
    grad_img = grad_img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+70,0:144]
    fig, ax = plt.subplots() # create a figure
    plt.imshow(final_img, cmap=plt.cm.gray) # show the image in greyscale
    plt.scatter(i_x, -i_y+mu_s[500]+70, marker='o', s=(73./fig.dpi)**2) # plot the detected gradient onto the image
    plt.plot((X)/(pix2mm)-0.5, -mu_s+mu_s[500]+70, 'r-', linewidth=0.5) # plot the interface fit
    plt.axis('off') # disable the showing of the axis
    Name=Fol_Out+ os.sep +'Step_'+str(idx)+'.png' # set output name
    MEX= 'Exporting Im '+ str(k)+' of ' + str(IMG_AMOUNT) # update on progress
    print(MEX) 
    plt.title('Image %04d' % ((idx-1121+1))) # set image title
    plt.savefig(Name, dpi= 400) # save image
    plt.close(fig) # disable or enabel depending on whether you want to see image in the plot window

# animate the result