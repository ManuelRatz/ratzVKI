# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:59:47 2020
@author: koval
@description: All functions for the edge detection.
"""
###############################################################################
# Functions
###############################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os
import cv2
import imageio
from scipy.signal import find_peaks
import Fitting_tool_functions as aFitting
from scipy.ndimage import gaussian_filter

def detect_triggersHS(folder):
    
    def read_lvm(path):
    
    #path_cam = folder + 'test-THS-cam_001.lvm'
        header    = 12  # number of header rows
        value     = 15  # gives the number of the loop
        with open(path) as alldata:
            line = alldata.readlines()[14]
        n_samples = int(line.strip().split('\t')[1])
    
        t = []
        v = []
        for i in range(value):
            with open(path) as alldata:                       #read the data points with the context manager
                lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
            t_temp        = [float(line.strip().split('\t')[0]) for line in lines] 
            v_temp        = [float(line.strip().split('\t')[1]) for line in lines]
            t             = np.append(t,t_temp)
            v             = np.append(v,v_temp)
        
        return t,v
    
    # paths
    path_cam = folder + '/test-HS-cam_001.lvm'
    t_cam,v_cam = read_lvm(path_cam)
    path_pr = folder + '/test-HS-pressure_001.lvm'
    t_pr,v_pr = read_lvm(path_pr)
    path_valve = folder + '/test-HS-valve_001.lvm'
    t_valve,v_valve = read_lvm(path_valve)
    
    
    f_acq = int(len(t_cam)/t_cam[-1])
    idx_start_cam = np.argwhere(v_cam > 3)[0]
    idx_start_valve = np.argwhere(v_valve > 4)[0]
    pressure_signal = v_pr[int(idx_start_valve):]
    frame0 = (t_valve[idx_start_valve]-t_cam[idx_start_cam])*f_acq
    return frame0, pressure_signal

def load_image(name, crop_index, idx, load_idx):
    """
    Function to load the image and highpass filter them for edge detection.

    Parameters
    ----------
    name : str
        Data path and prefix of the current run.
    crop_index : tuple of int
        4 coordinates to crop the images to.
    idx : int
        Current step counting from the beginning index.
    load_idx : int
        Current step counting from 0.

    Returns
    -------
    img_hp : 2d np.array of int
        Highpass filtered image.

    """
    # set the image name
    Image_name = name + '%05d' % load_idx + '.png'
    # load the image
    img=cv2.imread(Image_name,0)
    # crop the image
    img = img[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]
    # at the beginning we need to cheat a little with cv2 because the highpass
    # filtered image alone is not enough so we do a little denoising for the 
    # first 10 steps
    if (idx < 10):
        img = cv2.fastNlMeansDenoising(img.astype('uint8'),2,2,7,21)
    # convert the image to integer to avoid bound errors
    img = img.astype(np.int)
    # calculate the blurred image
    blur = gaussian_filter(img, sigma = 7, mode = 'nearest', truncate = 3)
    # subtract the blur yielding the interface
    img_hp = img - blur
    # subtract values below 3, this is still noise, so we kill it
    img_hp[img_hp<3] = 0
    # return the image
    return img_hp

def edge_detection_grad(crop_img, threshold_pos, wall_cut, threshold_outlier, kernel_threshold, do_mirror):
    """
    Function to get the the edges of the interface

    Parameters
    ----------
    crop_img : 2d np.array
        Image as array in grayscale.
    treshold_pos : float64
        Required threshold for the gradient to be taken as a valid value.
    wall_cut : int
        How many pixels to cut near the wall.
    threshold_outlier : float64
        Percentage value indicating the relative difference between y and the mean at which to filter.
    do_mirror : boolean
        If True, the right side of the image will be mirrored because the gradient is stronger there.

    Returns
    -------
    grad_img : 2d np.array
        Image as a binary map indicating the edge as 1.
    y_index : 1d np.array
        Y-coordinates of the edge.
    x_index : 1d np.array
        X-coordinates of the edge.

    """
    grad_img = np.zeros(crop_img.shape)
    y_index  = (np.asarray([])) 
    
    Profiles = crop_img[:,wall_cut:len(crop_img[1])-wall_cut] # analyse the region where no spots from the wall disturb detection
    # Profiles_s = savgol_filter(Profiles, 15, 2, axis =0)        # intensity profile smoothened
    Profiles_d = np.gradient(Profiles, axis = 0)              # calculate gradient of smoothend intensity along the vertical direction
    idx_maxima = np.zeros((Profiles.shape[1]),dtype = int)
    for i in range(0,Profiles_d.shape[1]):
        # if i == 132:
        #     print('Stop')
        if np.max(Profiles_d[:,i]) <= threshold_pos:
            idx_maxima[i] = int(np.argmax(Profiles_d[5:,i]))+5
        else:
            idx_maxima[i] = int(np.argmax(Profiles_d[5:,i] > threshold_pos))+5             # find positions of all the maxima in the gradient
    # idx_maxima = np.argmax(Profiles_d > 10, axis = 0)                # find positions of all the maxima in the gradient
    #mean_a = np.mean(Profiles_s, axis=0)
    for j in range(Profiles_d.shape[1]):
        # accept only the gradient peaks that are above the threeshold (it avoids false detections lines where the interface is not visible)  
        if Profiles_d[idx_maxima[j],j] > 1:
            grad_img[idx_maxima[j],j+wall_cut] = 1 # binarisation
        # else:
            # print('Below Threshold')
    if do_mirror == True:
        grad_img = mirror_right_side(grad_img)     
    y_index, x_index = np.where(grad_img==1) # get coordinates of the interface
    # sort both arrays to filter out outliers at the edges
    x_sort_index = x_index.argsort()
    y_index = y_index[x_sort_index[:]]
    x_index = x_index[x_sort_index[:]]

    # filter out outliers
    y_average  = np.median(y_index)
    for k in range(len(y_index)):
        kernel_size = 2 # amount of points to sample for median
        y_kernel=get_kernel(k, y_index,kernel_size)
        if np.abs(y_index[k]-np.median(y_kernel))/np.median(y_kernel) > kernel_threshold:
            grad_img[int(y_index[k]),x_index[k]] = 0
        if np.abs(y_index[k]-y_average)/np.median(y_index)>threshold_outlier:
            grad_img[int(y_index[k]),x_index[k]] = 0
    
    return grad_img, y_index, x_index

def mirror_right_side(array):
    """
    Function to mirror the right side of the image

    Parameters
    ----------
    array : 2d np.array
        Image to be mirrored.

    Returns
    -------
    mirrored : 2d np.array
        Mirrored image.

    """
    # take the right side of the array
    right = array[:,array.shape[1]//2+array.shape[1]%2:]
    # check if the width in pixels is even, if yes return the mirror
    if (array.shape[1]%2 == 0):
        mirrored = np.hstack((np.fliplr(right), right))
        # return the mirrored array
        return mirrored
    # if not we have to take the middle column
    middle = np.expand_dims(array[:,array.shape[1]//2], 1)
    # and add the right side to it, once normal and once flipped
    mirrored = np.hstack((np.fliplr(right), middle,right))
    # return the mirrored array
    return mirrored

def saveTxt(Fol_Out,h_mm, h_cl_l, h_cl_r, angle_l, angle_r):                
    """
    Function to save the calculated arrays in .txt files

    Parameters
    ----------
    Fol_Out : str
        Data path to the folder in which to store the images.
    h_mm : 1d np.array
        Average height of the meniscus in mm.
    h_cl_l : 1d np.array
        Left height of the contact angle in mm.
    h_cl_r : 1d np.array
        Right height of the contact line in mm.
    angle_l : 1d np.array
        Left contact angle in radians.
    angle_r : 1d np.array
        Right contact angle in Radians.
    """
    # create the folder if it doesn't exist
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)
    # save the txts, convert the angle to degrees
    np.savetxt(Fol_Out + os.sep + 'Disp_avg.txt',h_mm)
    np.savetxt(Fol_Out + os.sep + 'Disp_left.txt',h_cl_l)
    np.savetxt(Fol_Out + os.sep + 'Disp_right.txt',h_cl_r)
    np.savetxt(Fol_Out + os.sep + 'LCA.txt',angle_l*np.pi/180)
    np.savetxt(Fol_Out + os.sep + 'RCA.txt',angle_r*np.pi/180)

def get_kernel(k,y_index,kernel_size):
    """
    Function to get a defined kernel of an array.
    
    Parameters
    ----------
    k : int
        Index of the current iteration.
    y_index : 1d nupmy array
        Indices of the gradients.
    kernel_size : int
        Half the kernel length.

    Returns
    -------
    y_index : 1d numpy array
        Sliced index array of size 2*kernel_size+1.

    """
    if(k > kernel_size and k < len(y_index)-kernel_size): # points in the middle
        return y_index[k-kernel_size:k+kernel_size+1]
    elif(k <= kernel_size): # points at the left edge
        return y_index[0:2*kernel_size+1] 
    elif(k >= len(y_index)-kernel_size): # points at the right edge
        return y_index[len(y_index)-2*kernel_size-1:len(y_index)]
    else: # error message to be sure
        print('Error')
        return

def fitting_cosh(x_data, y_data):
    """
    Function to fit the experimental data to a hyperbolic cosine. The data has
    to be already extracted from the image to only be x and y values

    Parameters
    ----------
    x_data : 1d np.array
        X values of the channel centered around 0, not normalized.
    y_data : 1d np.array
        Y values of the detected interface, mean subtracted.

    Returns
    -------
    y_fit : 1d np.array
        Fitted y values of the detected interface, mean subtracted.

    """
    # define the help function
    def func(x_func,a,b):
        return np.cosh(np.abs(x_func)**a/b)-1
    # calculate the values of the fit
    popt_cons, _ = curve_fit(func, x_data, y_data, bounds = ([-np.inf,-np.inf],[np.inf,np.inf]))
    # calculate the fitted data
    y_fit = func(x_data, popt_cons[0], popt_cons[1])
    # return it
    return y_fit

def fitting_advanced(grad_img,pix2mm,l,sigma_f,sigma_y):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    # i_x = i_x-0.5
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = grad_img.shape[1]*pix2mm # width of the image in mm 
    
    X_t=i_x_mm; X_train=np.expand_dims(X_t, axis=1)
    Y_t=i_y_mm; Y_train=np.expand_dims(Y_t, axis=1)
    #y_c=data['y_c'] # This is the clean data.
    X=np.linspace(0, 5, 1000, endpoint=True)
    X = X-0.5*pix2mm
    X_test=np.expand_dims(X, axis=1)
    # Note that the vectors are augmented to be of sixe n x 1
    
    # This is the Gaussian Fit
    mu_s, cov_s = aFitting.posterior_predictive(X_test, X_train, Y_train, img_width_mm,sigma_f,sigma_y)
    mu_s = mu_s[:,0]
    # This is an example of sampling possibe solutions of the regressions
    #samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    
    # fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
    # plt.scatter(X_train,Y_train,c='white',
    #             marker='o',edgecolor='black',
    #             s=10,label='Data')
    # aFitting.plot_gp(mu_s, cov_s, X_test)
    # plt.xlim([0,5])
    # plt.ylim([30,40])
    

    return mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm

def vol_average(y,x,img_width_mm):
    """
    Function to calculate average height of the meniscus
    :grad_img: binary image, 1 at the edge of the meniscus
    :x: x coordinate of data for polynomial fitting
    :f: coordinates of the fitted curve
    
    """
    if len(y)==0:
        h_mm = np.nan
    else:
        vol = np.trapz(y, x)  # integrating the fitted line in 2D (?)
        h_mm = vol/(img_width_mm) 
    return h_mm


def contact_angle(y,x,side):
    """
    Function to calculate the contact angle
    :x: x coordinate of data for polynomial fitting
    :f: coordinates of the fitted curve
    :side: 0 for left side, -1 for right side
    :angle: initialisation for array
    
    """
    if side == 0:
        if np.gradient(y,x)[side]< 0:
            grad = abs(np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90-alfa_deg
        else:
            grad = (np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90+alfa_deg
    else:
        if np.gradient(y,x)[side]> 0:
            grad = abs(np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90-alfa_deg
        else:
            grad = (np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90+alfa_deg
    #angle_rad = np.radians(angle)
    return angle