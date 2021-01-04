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
import image_processing_functions as imp # for interface detection
from scipy.ndimage import gaussian_filter

FPS = 500 # camera frame rate in frame/sec
THRESHOLD = 12  # threshold for the gradient 
OUTL_THR = 0.3 # threshold for outliers
WALL_CUT = 2 # Cutting pixels near the wall
crop_index = (66,210,0,1280)
WIDTH = 5 # width of the channel
PIX2MM = WIDTH/(crop_index[1]-crop_index[0]) # pixel to mm

# name of the test case and the folder location
TESTNAME = '1500_A05000.png' # prefix of the images
Fol_In = 'G:\Anna\WATER\\20200825\\1500Pa\\test1\images'
NAME = Fol_In + os.sep + TESTNAME # file name


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

grad_img,y_index, x_index = imp.edge_detection_grad(dst,THRESHOLD,WALL_CUT,OUTL_THR, do_mirror = True) # calculate the position of the interface
mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(grad_img,PIX2MM,l=5,sigma_f=0.1,sigma_y=0.5e-6) # fit a gaussian

# fit the data to a cosine
mu_s = mu_s[:,0] # shape the array to match the x values
X_c = X - 2.5 + 0.5*PIX2MM # shift the x values to be centered around 0
subt = np.min(mu_s) # min subtraction of the y-values
Y_c = mu_s - subt # subtract the minimum
Y_fit = imp.fitting_cosh(X_c, Y_c) # fit to a cosine
Y_c += subt # add the min again
Y_fit += subt # add the min again
# plot to compare the two fits
fig, ax = plt.subplots()
ax.plot(X_c, Y_c, label = 'Raw')
ax.plot(X_c, Y_fit, label = 'Fit')
ax.legend(loc = 'upper center', ncol = 2)
ax.set_xlim(-2.5, 2.5)

# plot the result
mu_s = mu_s/PIX2MM # calculate the resulting height in mm
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
