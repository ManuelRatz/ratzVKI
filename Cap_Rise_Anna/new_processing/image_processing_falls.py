# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 07:37:07 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np # for array calculations
import cv2 # for image processing
from matplotlib import pyplot as plt # for plotting
import os # for filepaths
import image_processing_functions as imp # for interface detection
import post_processing_functions as ppf
import imageio

ppf.set_plot_parameters(20, 15, 10)


"""General settings"""
# amount of images to process
Img_Amount = 100

# width of the channel
Width = 5
# acquisition frequency of the camera
Fps = 500

"""Interface Detection"""
# threshold for the gradient 
Threshold_Gradient = 5
# threshold for outliers
Threshold_Outlier_out = 60 # 0.95
Threshold_Outlier_in = 20 # 0.3
# threshold for the kernel filtering
Threshold_Kernel_out = 3 #0.05
Threshold_Kernel_in = 2 #0.01

# whether to mirror the right side onto the left
Do_Mirror = True
Denoise = True

"""locate the images"""
# letter of the current run
Test_Case = 'slow_C'
Case = 'Fall'
Fluid = 'HFE'
plus = 0
Fol_Data, Pressure, Run, H_Final, Frame0, Crop_Index, Speed =\
    imp.get_parameters(Test_Case, Case, Fluid)
if Case == 'Rise':
    N_T = Frame0+Img_Amount-1+plus
else:
    N_T = Img_Amount = Frame0
    # plus = Frame0-50
    Frame0 = 0
    # Img_Amount = 1
if Img_Amount < 20:
    dpi = 400
else:
    dpi = 80
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 
# name of the output folder
Fol_Images_Detected = ppf.create_folder(Fol_Data + os.sep + 'Images_Detected')

"""initialize arrays"""
h_mm_avg = np.zeros([N_T])*np.nan  
h_cl_left = np.zeros([N_T])*np.nan  
h_cl_right = np.zeros([N_T])*np.nan  
angle_gauss = np.zeros([N_T])*np.nan  
angle_cosh = np.zeros([N_T])*np.nan 
curvature_gauss = np.zeros(([N_T]))
curvature_cosh = np.zeros(([N_T]))

def func(x_func,a,b):
    return (np.cosh(np.abs(x_func)**a/b)-1)

"""gif setup"""
images = [] # empty list to append into
GIFNAME = Test_Case + '_' + Case + '_' + Fluid + '.gif' # name of the gif
import time
start = time.time()

# loop over all the images
for i in range(0+plus,Img_Amount+plus):
    if (Img_Amount+plus-i)<30 and Fluid == 'HFE':
        Wall_Cut = 8
    else:
        # pixels to cut near the wall
        Wall_Cut = 3
    # get the index starting from N_T
    Idx = i
    # get the index starting from 0
    Load_Idx = Frame0+Idx
    MEX= 'Exporting Im '+ str(i+1)+' of ' + str(Img_Amount) # update on progress
    print(MEX) 
    # load the image and highpass filter it
    img_hp, img = imp.load_image(Fol_Data, Crop_Index, Idx, Load_Idx,\
                                 Pressure, Run, Speed, Denoise, Fluid)
    # calculate the detected interface position
    grad_img, y_index, x_index = imp.edge_detection_grad(img_hp,\
       Threshold_Gradient, Wall_Cut, Threshold_Outlier_in, Threshold_Outlier_out
       , Threshold_Kernel_out, Threshold_Kernel_in,\
           do_mirror = Do_Mirror, fluid = Fluid, idx = Idx)
    # fit a gaussian to the detected interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(\
        grad_img, Pix2mm, l=5, sigma_f=1e-3, sigma_y=1e-8)
    
    # this is the part with the hyperbolic cosine fit
    i_x_fit = i_x_mm - 2.5
    X_plot = (np.linspace(-2.5,2.5,1000))
    
    fit_val, shift, inverted = imp.fitting_cosh2(i_x_fit, i_y)

    y_cosh_fine = (func(X_plot, fit_val[0], fit_val[1]))
    if inverted == True:
        y_cosh_fine *= (-1)
    y_cosh_fine += shift
    y_cosh_fine *= Pix2mm
    # save the values into the arrays
    h_mm_avg[Load_Idx] = imp.vol_average(mu_s,X,img_width_mm)
    h_cl_left[Load_Idx] = mu_s[0]
    h_cl_right[Load_Idx] = mu_s[-1]
    angle_gauss[Load_Idx]= imp.contact_angle(mu_s,X,-1, Pix2mm)
    angle_cosh[Load_Idx]= imp.contact_angle(y_cosh_fine,X,-1, Pix2mm)
    curvature_gauss[Load_Idx] = imp.integrate_curvature(mu_s)
    curvature_cosh[Load_Idx] = imp.integrate_curvature(y_cosh_fine)
    
    # plot the result
    fig, ax = plt.subplots() # create a figure
    mu_s = mu_s/Pix2mm # convert back to pixels
    y_cosh_fine = y_cosh_fine/Pix2mm
    ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    ax.scatter(i_x, i_y, marker='x', s=(70./fig.dpi)**2, color = 'lime') # plot the detected gradient onto the image
    ax.plot((X)/(Pix2mm), mu_s, 'r-', linewidth=0.5) # plot the interface fit
    ax.plot((X_plot+2.5)/Pix2mm-0.5, y_cosh_fine, lw=0.5, color = 'yellow')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off') # disable the showing of the axis
    ax.set_ylim(mu_s[500]-10, mu_s[500]+85)
    # ax.set_ylim(0,100)
    fig.tight_layout()
    NAME_OUT = Fol_Images_Detected + os.sep + 'Stp_%05d.png' %(Idx+Frame0)
    fig.savefig(NAME_OUT, dpi= dpi) # save image
    plt.close(fig) # disable or enable depending on whether you want to see image in the plot window
    if (i%2) == 0:
        images.append(imageio.imread(NAME_OUT))
imageio.mimsave(GIFNAME, images, duration = 0.10)


# pressure, f0 = imp.load_labview_files(Fol_Data, Test_Case)

# h_offset = np.mean(h_mm_avg[-50:])

# h_mm_avg = h_mm_avg[Frame0+plus:]+H_Final-h_offset
# h_cl_left = h_cl_left[Frame0+plus:]+H_Final-h_offset
# h_cl_right = h_cl_right[Frame0+plus:]+H_Final-h_offset
# angle_gauss = angle_gauss[Frame0+plus:]
# angle_cosh = angle_cosh[Frame0+plus:]
# curvature_gauss = curvature_gauss[Frame0+plus:]
# curvature_cosh = curvature_cosh[Frame0+plus:]
# pressure = pressure[Frame0-f0:Frame0-f0+Img_Amount]
# t = np.linspace(0,len(h_mm_avg)/Fps,len(h_mm_avg)+1)[Frame0-f0:]

imp.saveTxt_fall(Fol_Data, h_mm_avg, h_cl_right, angle_gauss, angle_cosh, Test_Case)
# imp.saveTxt(Fol_Data, h_mm_avg, h_cl_left, h_cl_right, angle_gauss,\
#             angle_cosh, pressure, curvature_gauss, curvature_cosh,\
#                 Test_Case)
print(time.time()-start)