# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:35:47 2020
@author: koval
@description: Detection of the meniscus shape and calculation of height and contact angle
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import Image_processing_functions as imgprocess
import imageio
###############################################################################
# Input parameters
###############################################################################
    
fps = 500 # camera frame rate in frame/sec

n_start = 1121# first image where the interface appears (check the images)
n_t = n_start+2000 # number of images


treshold_pos = 1  # threshold for the gradient 

wall_cut = 0 # Cutting pixels near the wall

crop_indx1 = 66  # first x coordinate to crop the image
crop_indx2 = 210 # second x coordinate to crop the image
crop_indy1 = 0 # first y coordinate to crop the image
crop_indy2 = 1280 # second y coordinate to crop the image

width = 5 # width of the channel
pix2mm = width/(crop_indx2-crop_indx1) # pixel to mm

testname = '350Pa_C'
date_exp = '2020-08-28'
folder = 'test_images' + os.sep
name = folder + os.sep + testname  # file name

Fol_Out= folder + os.sep + 'images_detected/'
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
    
###############################################################################
#%%  For loop to process every frame
###############################################################################


plt.close('all')

Img_amount = 5 # amount of images to process

t_exp = np.linspace(0,(n_t)/fps,(Img_amount))

# initialization
h_mm_adv = np.zeros([Img_amount])*np.nan  

h_cl_left_all_adv = np.zeros([Img_amount])*np.nan  
h_cl_right_all_adv = np.zeros([Img_amount])*np.nan  

angle_all_left_adv = np.zeros([Img_amount])*np.nan  
angle_all_right_adv = np.zeros([Img_amount])*np.nan  

images=[] 
# for k in range(n_exp,n_t+1): 
for k in range(0,5):
    idx = n_start+136+20*k
    image = name + '%05d' %idx + '.png'  # file name
    # print(image)
    img=cv2.imread(image,0)  # read the image
    dst = cv2.fastNlMeansDenoising(img,10,1,7,21)
    dst2 = 3*dst.astype(np.float64)
    crop_img =dst2[crop_indy1:crop_indy2,crop_indx1:crop_indx2]  # crop
    
    #Calculate gradient of intensity
    #--------------------------------------------------------------------------
    grad_img,y_index, x_index = imgprocess.edge_detection_grad(crop_img,treshold_pos,wall_cut)

    # Gaussian fitting
    #--------------------------------------------------------------------------
    #  l: Kernel length parameter.
    #  sigma_f: Kernel vertical variation parameter.
    #  sigma_y: Noise parameter.
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imgprocess.fitting_advanced(grad_img,pix2mm,l=5,sigma_f=2000,sigma_y=10)
    
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

    h_mm_adv[k] =h_A_mm_adv    #mm  # differences to equilibrium height

    h_cl_left_all_adv[k] =h_cl_left_adv    
    h_cl_right_all_adv[k] =h_cl_right_adv   
         
    # Calculate contact angle
    #--------------------------------------------------------------------------
   
    angle_all_left_adv[k]= angle_left_adv
    angle_all_right_adv[k]= angle_right_adv
    
    # Generate temporary images with the fitted curve
    #--------------------------------------------------------------------------
    mu_s = mu_s/pix2mm
    crop_img1 = crop_img[int(1280-mu_s[500])-60:int(1280-mu_s[500])+60,0:144]
    
    plt.figure()
    plt.imshow(crop_img1, cmap=plt.cm.gray)
    plt.plot((X)/(pix2mm)-0.5, -mu_s+mu_s[500]+60, 'r-', linewidth=0.5)
    #plt.plot(i_x,len(grad_img[:,0])-i_y,'x')
    plt.axis('off')
    Name=Fol_Out+ os.sep +'Step_'+str(idx)+'.png'
    MEX= 'Exporting Im '+ str(k+1)+' of ' + '5'
    print(MEX)
    # plt.grid()
    plt.title('Image %04d' % (idx-n_start+1))
    plt.savefig(Name)   
    images.append(imageio.imread(Name))
    # plt.close('all')

#%% Generate gif
GIFNAME='AnimationTest.gif'
imageio.mimsave(GIFNAME, images,duration=0.01)
# uncomment the next line for animation
# imgprocess.animation(GIFNAME,Fol_Out,n_t-1,n_start)

#%% Save variables

def saveTxt(Fol_Out,h_mm, h_cl_l, h_cl_r, angle_l, angle_r):                
    
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)
    np.savetxt(Fol_Out + os.sep + 'Displacement.txt',h_mm)
    
    np.savetxt(Fol_Out + os.sep + 'Displacement_CLsx.txt',h_cl_l)
    np.savetxt(Fol_Out + os.sep + 'Displacement_CLdx.txt',h_cl_r)
    
    np.savetxt(Fol_Out + os.sep + 'LCA.txt',angle_l*np.pi/180)
    np.savetxt(Fol_Out + os.sep + 'RCA.txt',angle_r*np.pi/180)
    
Fol_Out_Adv= os.path.abspath(folder +'/Txts_advanced_fitting')
if not os.path.exists(Fol_Out_Adv):
    os.mkdir(Fol_Out_Adv)
saveTxt(Fol_Out_Adv,h_mm_adv, h_cl_left_all_adv, h_cl_right_all_adv, angle_all_left_adv, angle_all_right_adv)