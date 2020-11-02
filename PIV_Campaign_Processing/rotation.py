#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:52:25 2020

@author: ratz
@description: functions for rotating the raw PIV images
"""

import cv2                  # for reading the images
from scipy import ndimage   # for rotating the images
import numpy as np          # for array operations
import os                   # for filepaths
import matplotlib.pyplot as plt # to check if the angle calculation is okay

def get_angle(image):
    """
    Function to calculate the angle of tilt

    Parameters
    ----------
    image : 2d array
        Greyscale image of the channel.

    Returns
    -------
    angle_left : float64
        Angle of tilt in radians.

    """
    def linear(a, b, x):
        """
        Help function for the linear fit of the images
        """
        return a*x+b
    peak_left, peak_right= find_edges(image) # get the indices of the detected edges
    x_dummy = np.arange(1,1281,1) # set up a dummy for the fit
    a,b = np.polyfit(x_dummy,peak_left,1) # 
    angle_left = np.abs(np.arctan((linear(a,b,1280)-b)/1280))
    return angle_left

def find_edges(image):
    """
    Function to find the edges of the channel by calculating the gradient.

    Parameters
    ----------
    image : 2d array
        Greyscale image of the channel.

    Returns
    -------
    peak_left : 1d Numpy array
        Indices of the detected wall on the left side.
    peak_right : 1d Numpy array
        Indices of the detected wall on the right side.

    """
    grad = np.gradient(image, axis = 1) # take the gradient of the rows
    peak_left = np.argmax(image>, axis = 1) # find the first peak from the left side
    peak_right = image.shape[1]-np.argmax(image[:,::-1]>100, axis =1) # find the first peak from the right side
    return peak_left, peak_right

def get_most_common_element(array):
    """
    Function to find the most common element in a numpy array

    Parameters
    ----------
    array : 1d Numpy array
        Array of interest.

    Returns
    -------
    idx : Integer
        Most common element in the array. In our case this is an index.

    """
    counts = np.bincount(array)
    idx = np.argmax(counts)    
    return idx

def rotate_and_crop(image, crop_final, crop_rot, angle):
    """
    Rotate and crop an image

    Parameters
    ----------
    image : 2d array
        Greyscale image of the channel.
    crop_final : Tuple of 4 integers
        Contains the final crop used to get just the channel in the image.
    crop_rot : Tuple of 4 integers
        Contains the Crop of the rotated image to cut the edges.
    angle : float64
        Angle of tilt for the image.

    Returns
    -------
    final_img : 2d array
        Final image for PIV processing, cropped and rotated accordingly.

    """
    rotated = ndimage.rotate(image, angle*180/np.pi)
    cropped = rotated[crop_rot[0]:crop_rot[1],crop_rot[2]:crop_rot[3]]
    final_img = cropped[crop_final[0]:crop_final[1],crop_final[2]:crop_final[3]]
    return final_img

def get_process_params(name, idx0, wallcut=0):
    """
    Function to fetch the required cropping variables for ALL the images.
    Should be checked because these values are used for every image

    Parameters
    ----------
    name : string
        Nomenclature of the images.
    idx0 : Integer
        Index for which to do the calculations.

    Returns
    -------
    crop_final : Tuple of 4 integers
        Contains the final crop used to get just the channel in the image.
    crop_rot : Tuple of 4 integers
        Contains the Crop of the rotated image to cut the edges.
    angle : float64
        Angle of tilt for the image.
    name_sliced : Tuple of 3 integers
        Indices of the "." in the name of the file
    """
    indices = [i for i, a in enumerate(name) if a == '.']
    name_sliced = name[:(indices[0]+1)]
    img_name = Fol_In + name + '%06d.tif' %idx0
    img = cv2.imread(img_name,0)
    # get the rotation angle and rotate
    angle = get_angle(img)
    rotated = ndimage.rotate(img, angle*180/np.pi)
    hr, wr = rotated.shape
    
    crop_rot =(0+int(np.rint(wr*np.tan(angle))), hr-int(np.rint(wr*np.tan(angle))),\
               0+int(np.rint(hr*np.tan(angle))), wr-int(np.rint(hr*np.tan(angle))))
    cropped = rotated[crop_rot[0]:crop_rot[1],crop_rot[2]:crop_rot[3]]
    
    new_peak_left, new_peak_right = find_edges(cropped)
    x_low =  get_most_common_element(new_peak_left)
    x_high = get_most_common_element(new_peak_right)
    crop_final = (0,cropped.shape[0],x_low+wallcut,x_high+1-wallcut)
    return crop_final, crop_rot, angle, name_sliced
    
    
# set up the input and output folder
Fol_In ='C:'+os.sep+'PIV_Campaign'+os.sep+'Rise\h1\p1000'+os.sep+'R_h1_f1000_1_p10'+os.sep
Fol_Out ='C:'+os.sep+'PIV_Campaign'+os.sep+'Rise\h1\p1000'+os.sep+'R_h1_f1000_1_p10_rotated'+os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)


# get the cropping parameters from the first image
first_frame = 1
name = 'R_h1_f1000_1_p10.6dbfz5ty.'
crop_final, crop_rot, angle, name_sliced = get_process_params(name, first_frame, wallcut=5)
img_list = os.listdir(Fol_In)
n_t = len(img_list)-5 # amount of images to calculate, take away the lvm files
# iterate over all of the images
for i in range(0, 5):
    img_name = name + '%06d.tif' %(i+first_frame) # get the image name
    img = cv2.imread(Fol_In + img_name,0) # read the image
    img_processed = rotate_and_crop(img, crop_final, crop_rot, angle) # rotate and crop the image
    name_out = name_sliced + '%06d.tif' %(i+first_frame) # crop the name for the output
    cv2.imwrite(Fol_Out + name_out ,img_processed) # write the cropped images to the output folder
    MEX = 'Cropping Image ' + str(i+1) + ' of ' + str(n_t)
    print(MEX)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    