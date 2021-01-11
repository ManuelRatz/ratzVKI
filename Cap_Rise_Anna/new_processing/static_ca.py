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
Img_Amount = 1
# width of the channel
Width = 5
# acquisition frequency of the camera
Fps = 500

"""Interface Detection"""
# threshold for the gradient 
Threshold_Gradient = 6
Threshold_Int = 8
# threshold for outliers
Threshold_Outlier = 0.2
# threshold for the kernel filtering
Threshold_Kernel= 0.02
# pixels to cut near the wall
Wall_Cut = 3 
# whether to mirror the right side onto the left
Do_Mirror = True
Denoise = True
"""locate the images"""
# letter of the current run
Test_Case = 'middle_B'
Case = 'Fall'
Fluid = 'Water'

Fol_Data, Pressure, Run, H_Final, Frame0, Crop_Index, Speed =\
    imp.get_parameters(Test_Case, Case, Fluid)
N_T = Frame0+Img_Amount-1
Pix2mm = Width/(Crop_Index[1]-Crop_Index[0]) 
# name of the output folder
Fol_Out = ppf.create_folder('Static_Ca')

def func(x_func,a,b):
        return (np.cosh(np.abs(x_func)**a/b)-1)

# loop over all the images
for i in range(0,Img_Amount):
    # get the index starting from N_T
    Idx = i
    # get the index starting from 0
    Load_Idx = Frame0+Idx+0
    MEX= 'Exporting Im '+ str(i+1)+' of ' + str(Img_Amount) # update on progress
    print(MEX) 
    # load the image and highpass filter it
    img_hp, img = imp.load_image(Fol_Data, Crop_Index, Idx, Load_Idx, Pressure, Run, Speed, Denoise)
    # calculate the detected interface position
    grad_img,y_index, x_index = imp.edge_detection_grad(img_hp,\
           Threshold_Gradient, Wall_Cut, Threshold_Outlier, Threshold_Kernel,
           Threshold_Int, do_mirror = Do_Mirror)
    # fit a gaussian to the detected interface
    mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm = imp.fitting_advanced(\
        grad_img ,Pix2mm, l=5, sigma_f=0.1, sigma_y=0.5e-6)
    imp.contact_angle(mu_s,X,0, Pix2mm)   
    X_c = X - 2.5 + 0.5*Pix2mm # shift the x values to be centered around 0
    subt = np.min(mu_s) # min subtraction of the y-values
    Y_c = mu_s - subt # subtract the minimum
    Y_fit = imp.fitting_cosh2(X_c, Y_c)
    
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
    
    # iterate
    for j in range(0, n_trials):
        # obtain training set
        xs, xss, ys, yss = train_test_split(x_to_fit, y_to_fit, test_size = 0.3)
        # fit with the training data
        fit_val = imp.fitting_cosh3(xs, ys)
        # calculate the fitting values
        a[j] = fit_val[0]; b[j] = fit_val[1]
        # calculate the y values on the fine grid
        y_reg[:,j] = func(X_c, fit_val[0], fit_val[1])
        # calculate the error
        e_out[j] = np.linalg.norm(yss-func(xss, fit_val[0], fit_val[1]))/np.sqrt(len(yss)-1)
    # calculate the mean y value
    y_reg_mean = np.mean(y_reg, axis = 1)
    y_reg_std = np.std(y_reg, axis = 1)
    uncertainty = 1.96*np.sqrt((e_out.mean()**2 + y_reg_std**2))/Pix2mm
    
    y_shift = (y_reg_mean+delta_i_y)/Pix2mm
    
    fig, ax = plt.subplots()
    ax.fill_between(X_c, uncertainty, -uncertainty, alpha=0.5)
    ax.set_xticks(np.linspace(-2.5, 2.5, 11))
    ax.set_ylabel('Uncertainty [mm]')
    ax.set_xlabel('$x$[mm]')
    ax.set_xlim(-2.5, 2.5)
    fig.tight_layout(pad= 1.1)
    fig.savefig(Fol_Out + os.sep + Test_Case +'_uncertainty.png', dpi = 500)
    
    Y_c += subt # add the min again
    Y_fit += subt # add the min again
    
    # plot to compare the two fits
    fig, ax = plt.subplots()
    ax.plot(X_c, (Y_c-np.min(Y_c))/np.max(Y_c-np.min(Y_c)), label = 'Raw')
    ax.plot(X_c, (Y_fit-np.min(Y_fit))/np.max(Y_fit-np.min(Y_fit)), label = 'Exponential Fit')
    # ax.legend(loc = 'upper center', ncol = 2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    ax.set_xticks(np.linspace(-2.5, 2.5, 11))
    fig.tight_layout()
    Name_Out = Fol_Out + os.sep + Test_Case + '_exponential_Fit.png'
    fig.savefig(Name_Out, dpi = 400)
    plt.close(fig)
    
    # plot the result with the interface detection
    fig, ax1 = plt.subplots() # create a figure
    # Y_fit += subt
    # Y_fit /= Pix2mm
    mu_s = mu_s/Pix2mm # convert back to pixels
    ax1.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    ax1.scatter(i_x, i_y, marker='x', s=(100./fig.dpi)**2, color = 'red') # plot the detected gradient onto the image
    # plot the exponential fit
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.axis('off') # disable the showing of the axis
    ax1.set_ylim(mu_s[500]-20, mu_s[500]+65)
    ax1.fill_between(X/Pix2mm, y_shift+uncertainty, y_shift-uncertainty, alpha=0.5)
    ax1.plot(X/Pix2mm, (y_shift), color = 'yellow')
    # ax2.set_xticks(np.linspace(-2.5, 2.5, 11))
    ax1.set_ylabel('Uncertainty [mm]')
    ax1.set_xlabel('$x$[mm]')
    ax1.set_xlim(0, 50)
    NAME_OUT = Fol_Out + os.sep + Test_Case + '_static_ca_fit.png'
    fig.tight_layout()
    fig.savefig(NAME_OUT, dpi= 400) # save image
    # plt.close(fig)
    
    # plot the result without the interface detection
    fig, ax = plt.subplots() # create a figure
    ax.imshow(np.flip(img, axis = 0), cmap=plt.cm.gray) # show the image in greyscale
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off') # disable the showing of the axis
    ax.set_ylim(mu_s[500]-20, mu_s[500]+65)
    NAME_OUT = Fol_Out + os.sep + Test_Case + '_static_ca_no_fit.png'
    fig.tight_layout()
    fig.savefig(NAME_OUT, dpi= 400) # save image
    plt.close(fig)