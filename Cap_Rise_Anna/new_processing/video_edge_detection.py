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
import Image_processing_functions as imgprocess # for interface detection
from skimage import exposure # for greyscaling and histogramm equalisation

FPS = 500 # camera frame rate in frame/sec
n_start = 1121# first image where the interface appears (check the images)
n_t = n_start+2000
THRESHOLD = 0.7  # threshold for the gradient 
Outlier_threshold = 0.10 # threshold for outliers
WALL_CUT = 3 # Cutting pixels near the wall
crop_index = (66, 210, 0, 1280) # Pixels to keep (xmin, xmax, ymin, ymax)
WIDTH = 5 # width of the channel
pix2mm = WIDTH/(crop_index[1]-crop_index[0]) # pixel to mm

testname = '350Pa_C' # prefix of the images
FOL = '..' + os.sep + '..' + os.sep + '..' + os.sep + 'test_images' + os.sep +\
    'raw_images' + os.sep # input folder
name = FOL + os.sep + testname  # file name

# create the output folder
Fol_Out= '..' + os.sep + '..' + os.sep + '..' + os.sep + 'test_images' + os.sep\
    + 'images_detected/' + os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

h_mm_adv = np.zeros([n_t+1])*np.nan  
h_cl_left_all_adv = np.zeros([n_t+1])*np.nan  
h_cl_right_all_adv = np.zeros([n_t+1])*np.nan  
angle_all_left_adv = np.zeros([n_t+1])*np.nan  
angle_all_right_adv = np.zeros([n_t+1])*np.nan  

IMG_AMOUNT = 2000 # amount of images to process
Image = 1492
# iterate over all images 
for k in range(0,IMG_AMOUNT+1):
    idx = n_start+k # get the first index
    image = name + '%05d' %idx + '.png'  # file name
    img=cv2.imread(image,0)  # read the image
    img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]  # crop
    dst = cv2.fastNlMeansDenoising(img,2,2,7,21) # denoise
    dst2 = 3*dst.astype(np.float64) # increase contrast
 
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(dst2,THRESHOLD,WALL_CUT,Outlier_threshold) # calculate the position of the interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,pix2mm,l=5,sigma_f=0.5,sigma_y=4e-5) # fit a gaussian
    
    # Calculate average height
    #--------------------------------------------------------------------------

    h_A_mm_adv = imgprocess.vol_average(mu_s[:,0],X,img_width_mm)
    
    # Calculate contact line height
    #--------------------------------------------------------------------------
    h_cl_left_adv = mu_s[0]
    h_cl_right_adv = mu_s[-1]
          
    # Calculate contact angle
    #--------------------------------------------------------------------------
    angle_left_adv = imgprocess.contact_angle(mu_s[:,0],X,0)
    angle_right_adv = imgprocess.contact_angle(mu_s[:,0],X,-1)

    h_mm_adv[k+n_start] =h_A_mm_adv    #mm  # differences to equilibrium height

    h_cl_left_all_adv[k+n_start] =h_cl_left_adv    
    h_cl_right_all_adv[k+n_start] =h_cl_right_adv   
         
    # Calculate contact angle
    #--------------------------------------------------------------------------
    angle_all_left_adv[k+n_start]= angle_left_adv
    angle_all_right_adv[k+n_start]= angle_right_adv
    
    mu_s = mu_s/pix2mm # calculate the resulting height in mm
    
    # # plot the result
    # final_img = img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+50,0:144]
    # grad_img = grad_img[int(1280-mu_s[500])-70:int(1280-mu_s[500])+50,0:144]
    # fig, ax = plt.subplots() # create a figure
    # plt.imshow(final_img, cmap=plt.cm.gray) # show the image in greyscale
    # plt.scatter(i_x, -i_y+mu_s[500]+70, marker='o', s=(73./fig.dpi)**2) # plot the detected gradient onto the image
    # plt.plot((X)/(pix2mm)-0.5, -mu_s+mu_s[500]+70, 'r-', linewidth=0.5) # plot the interface fit
    # plt.axis('off') # disable the showing of the axis
    # Name=Fol_Out+ os.sep +'Step_'+str(idx)+'.png' # set output name
    MEX= 'Exporting Im '+ str(k)+' of ' + str(IMG_AMOUNT) # update on progress
    print(MEX) 
    # plt.title('Image %04d' % ((idx-1121+1))) # set image title
    # plt.savefig(Name, dpi= 100) # save image
    # plt.close(fig) # disable or enabel depending on whether you want to see image in the plot window

# animate the result

def saveTxt(Fol_Out,h_mm, h_cl_l, h_cl_r, angle_l, angle_r):                
    
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)
    np.savetxt(Fol_Out + os.sep + 'Displacement.txt',h_mm)
    
    np.savetxt(Fol_Out + os.sep + 'Displacement_CLsx.txt',h_cl_l)
    np.savetxt(Fol_Out + os.sep + 'Displacement_CLdx.txt',h_cl_r)
    
    np.savetxt(Fol_Out + os.sep + 'LCA.txt',angle_l*np.pi/180)
    np.savetxt(Fol_Out + os.sep + 'RCA.txt',angle_r*np.pi/180)
    
Fol_Out_Adv= os.path.abspath(FOL + os.sep + 'Txts_advanced_fitting')
if not os.path.exists(Fol_Out_Adv):
    os.mkdir(Fol_Out_Adv)
saveTxt(Fol_Out_Adv,h_mm_adv, h_cl_left_all_adv, h_cl_right_all_adv, angle_all_left_adv, angle_all_right_adv)