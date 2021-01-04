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

FPS = 500 # camera frame rate in frame/sec
# tests = ['test1','test2','test3']
N_START = 1121# first image where the interface appears (check the images)
N_T = N_START+2000
THRESHOLD = 10  # threshold for the gradient 
OUTL_THR = 0.3 # threshold for outliers
WALL_CUT = 2 # Cutting pixels near the wall
crop_index = (66,210,0,1280) # Pixels to keep (xmin, xmax, ymin, ymax)
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

IMG_AMOUNT = N_T # amount of images to process
plus = 0

images = []
GIFNAME = 'Detected_interface.gif'
# iterate over all images 
for k in range(0,1):
    idx = N_START+1*k+plus # get the first index
    image = NAME + '%05d' %idx + '.png'  # file name
    img=cv2.imread(image,0)  # read the image
    img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]  # crop
    dst = cv2.fastNlMeansDenoising(img,2,2,7,21) # denoise
    dst2 = 3*dst.astype(np.float64) # increase contrast
 
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(dst,THRESHOLD,WALL_CUT,OUTL_THR, do_mirror = True) # calculate the position of the interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,PIX2MM,l=5,sigma_f=0.1,sigma_y=0.5e-6) # fit a gaussian
    mu_s = mu_s[:,0]
    # h_mm_adv[idx] = imgprocess.vol_average(mu_s[:,0],X,img_width_mm)    #mm  # differences to equilibrium height
    # h_cl_left_all_adv[idx] = mu_s[0]
    # h_cl_right_all_adv[idx] = mu_s[-1]
    # angle_all_left_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,0)
    # angle_all_right_adv[idx]= imgprocess.contact_angle(mu_s[:,0],X,-1)
    
    mu_s = mu_s/PIX2MM # calculate the resulting height in mm

    # plot the result
    final_img = img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+50,0:500]
    grad_img = grad_img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+50,0:500]
    fig, ax = plt.subplots() # create a figure
    plt.imshow(final_img, cmap=plt.cm.gray) # show the image in greyscale
    plt.scatter(i_x, -i_y+mu_s[500]+70, marker='x', s=(13./fig.dpi)**2) # plot the detected gradient onto the image
    plt.plot((X)/(PIX2MM), -mu_s+mu_s[500]+70, 'r-', linewidth=0.5) # plot the interface fit
    # plt.plot((X2)/(PIX2MM)-0.5, -mu_s2+mu_s2[500]+70, 'y-', linewidth=0.5) # plot the interface fit
    plt.axis('off') # disable the showing of the axis
    NAME_OUT=Fol_Out+ os.sep +'Step_'+str(idx)+'.png' # set output name
    MEX= 'Exporting Im '+ str(idx+1)+' of ' + str(IMG_AMOUNT) # update on progress
    print(MEX) 
    plt.title('Image %04d' % ((idx-1121))) # set image title
    # plt.savefig(NAME_OUT, dpi= 500) # save image
    # plt.close(fig) # disable or enable depending on whether you want to see image in the plot window
    images.append(imageio.imread(NAME_OUT))

# imageio.mimsave(GIFNAME, images, duration = 0.05)
# def saveTxt(Fol_Out,h_mm, h_cl_l, h_cl_r, angle_l, angle_r):                
    
#     if not os.path.exists(Fol_Out):
#         os.mkdir(Fol_Out)
#     np.savetxt(Fol_Out + os.sep + 'Displacement.txt',h_mm)
    
#     np.savetxt(Fol_Out + os.sep + 'Displacement_CLsx.txt',h_cl_l)
#     np.savetxt(Fol_Out + os.sep + 'Displacement_CLdx.txt',h_cl_r)
    
#     np.savetxt(Fol_Out + os.sep + 'LCA.txt',angle_l*np.pi/180)
#     np.savetxt(Fol_Out + os.sep + 'RCA.txt',angle_r*np.pi/180)
    
# Fol_Out_Adv= os.path.abspath(Fol_Out + os.sep + 'Txts_advanced_fitting')
# if not os.path.exists(Fol_Out_Adv):
#     os.mkdir(Fol_Out_Adv)
# saveTxt(Fol_Out_Adv,h_mm_adv, h_cl_left_all_adv, h_cl_right_all_adv, angle_all_left_adv, angle_all_right_adv)