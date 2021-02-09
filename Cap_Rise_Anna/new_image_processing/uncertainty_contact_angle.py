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
from sklearn.model_selection import train_test_split


ppf.set_plot_parameters(20, 15, 10)

"""General settings"""
# amount of images to process
Img_Amount = 2000
# width of the channel
Width = 5
# acquisition frequency of the camera
Fps = 500

"""Interface Detection"""
# threshold for the gradient 
Threshold_Gradient = 0
# threshold for outliers
Threshold_Outlier_out = 60 # 0.95
Threshold_Outlier_in = 20 # 0.3
# threshold for the kernel filtering
Threshold_Kernel_out = 2 #0.05
Threshold_Kernel_in = 2 #0.01
# pixels to cut near the wall
Wall_Cut = 4
# whether to mirror the right side onto the left
Do_Mirror = True
Denoise = True

"""locate the images"""
# letter of the current run
Test_Case = 'P1500_C30_A'
Case = 'Rise'
Fluid = 'Water'

Fol_Data, Pressure, Run, H_Final, Frame0, Crop_Index, Speed =\
    imp.get_parameters(Test_Case, Case, Fluid)
N_T = Frame0+Img_Amount-1
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 
# name of the output folder
Fol_Out = ppf.create_folder('Crop_Check')
Fol_Images_Detected = ppf.create_folder(Fol_Data + os.sep + 'Images_Detected_Unc')

"""initialize arrays"""
h_mm_avg = np.zeros([N_T+1])*np.nan  
h_cl_left = np.zeros([N_T+1])*np.nan  
h_cl_right = np.zeros([N_T+1])*np.nan  
angle_left = np.zeros([N_T+1])*np.nan  
angle_right = np.zeros([N_T+1])*np.nan 
fit_coordinates = np.zeros((1000,Img_Amount))


"""gif setup"""
images = [] # empty list to append into
GIFNAME = 'Detected_interface.gif' # name of the gif
fit_val = [1,1,1,1]
# loop over all the images
i = 180
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
          Threshold_Gradient, Wall_Cut, Threshold_Outlier_in, Threshold_Outlier_out,
          Threshold_Kernel_out, Threshold_Kernel_in,
          do_mirror = Do_Mirror, fluid = Fluid, idx = Idx)
# fit a gaussian to the detected interface
mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(\
    grad_img ,Pix2mm, l=5, sigma_f=1e-3, sigma_y=1e-8)
    
# this is the part with the hyperbolic cosine fit
i_x_fit = i_x_mm - 2.5
X_plot = (np.linspace(-2.5,2.5,1000))

fit_val, shift, inverted = imp.fitting_cosh2(i_x_fit, i_y)
def func(x_func,a,b):
        return (np.cosh(np.abs(x_func)**a/b)-1)
y_cosh_fine = (func(X_plot, fit_val[0], fit_val[1]))
if inverted == True:
    y_cosh_fine *= (-1)
y_cosh_fine += shift
y_cosh_fine *= Pix2mm

X_c = X - 2.5 + 0.5*Pix2mm # shift the x values to be centered around 0

# number of simulations
n_trials = 100
# matrix to store the y values
y_reg = np.zeros((X_c.shape[0], n_trials))
# vector to store the errors in
e_out = np.zeros((n_trials,1))
# values of a and b for each simulation
a = np.zeros((n_trials, 1)); b = np.zeros((n_trials, 1))
# data to fit
x_to_fit = i_x_mm - 2.5
delta_i_y = np.min(i_y_mm)
y_to_fit = i_y_mm-delta_i_y

for j in range(0, n_trials):
    # obtain training set
    xs, xss, ys, yss = train_test_split(x_to_fit, i_y_mm, test_size = 0.3)
    fit_val, shift, inverted = imp.fitting_cosh2(xs, ys) # fit to a cosine    
    # calculate the y values on the fine grid
    a[j] = fit_val[0]; b[j] = fit_val[1]
    y_reg[:,j] = func(X_c, fit_val[0], fit_val[1])
    e_out[j] = np.linalg.norm(yss-(func(xss, fit_val[0], fit_val[1])+shift))/np.sqrt(len(yss)-1)
# calculate the mean y value
y_reg_mean = np.mean(y_reg, axis = 1)
y_reg_std = np.std(y_reg, axis = 1)
uncertainty = 1.96*np.sqrt((e_out.mean()**2 + y_reg_std**2))/Pix2mm

y_cosh_fine = y_cosh_fine/Pix2mm
mu_s = mu_s/Pix2mm # convert back to pixels

fig, ax1 = plt.subplots() # create a figure
ax1.invert_yaxis()
ax1.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
# ax1.scatter(i_x, i_y, marker='x', s=(100./fig.dpi)**2, color = 'red') # plot the detected gradient onto the image
# ax1.plot(X/Pix2mm, mu_s, color = 'lime', lw = 0.4)
# plot the exponential fit
ax1.set_aspect('equal')
ax1.axis('off') # disable the showing of the axis
ax1.fill_between(X/Pix2mm, y_cosh_fine+uncertainty, y_cosh_fine-uncertainty, alpha=0.6)
ax1.plot(X/Pix2mm, y_cosh_fine, color = 'yellow', lw = 0.4)
ax1.set_ylabel('Uncertainty [mm]')
ax1.set_xlabel('$x$[mm]')
ax1.set_xlim(img.shape[1]//2, img.shape[1])
ax1.set_ylim(mu_s[500]-20, mu_s[500]+65)
fig.tight_layout()
fig.savefig('test.png', dpi = 400)
# NAME_OUT = Fol_Images_Detected + os.sep + 'Imageglob_%05d.png' %Idx
# fig.tight_layout()
    # fig.savefig(NAME_OUT, dpi= 80) # save image
    # images.append(imageio.imread(NAME_OUT))
    # plt.close(fig)
# imageio.mimsave(GIFNAME, images, duration = 0.10)

# pressure, f0 = imp.load_labview_files(Fol_Data, Test_Case)

# fig = plt.figure()
# ax = plt.gca()
# ax.fill_between(X/Pix2mm, uncertainty, -uncertainty, alpha=0.6)


# h_mm_avg = h_mm_avg[Frame0:]
# h_cl_left = h_cl_left[Frame0:]
# h_cl_right = h_cl_right[Frame0:]
# angle_left = angle_left[Frame0:]
# angle_right = angle_right[Frame0:]
# pressure = pressure[Frame0-f0:Frame0-f0+Img_Amount]
# t = np.linspace(0,len(h_mm_avg)/Fps,len(h_mm_avg)+1)[Frame0-f0:]

# imp.saveTxt(Fol_Data, h_mm_avg, h_cl_left, h_cl_right, angle_left,\
#             angle_right, pressure, fit_coordinates, Test_Case)