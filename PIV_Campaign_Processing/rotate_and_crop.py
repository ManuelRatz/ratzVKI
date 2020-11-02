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
    # plt.scatter(x_dummy, peak_left, marker='o', s=(73./100)**2)
    # plt.plot(x_dummy,linear(a,b,x_dummy))
    # plt.show()
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
    peak_left = np.argmax(image>160, axis = 1) # find the first peak from the left side
    peak_right = image.shape[1]-np.argmax(image[:,::-1]>160, axis =1) # find the first peak from the right side
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

def get_process_params(folder):
    """
    Function to get the rotation angle and crop range

    Parameters
    ----------
    folder : string
        Input folder where the images are located.

    Returns
    -------
    crop_final : integer array
        Array containing the 4 crop parameters at the end.
    crop_rot : integer array
        Array containing the 4 crop parameters after the rotation.
    angle : float64
        Angle of tilt for the images.
    name_sliced : string
        Name of the images without the number.
    img_amount : integer
        Amount of images in the folder.
    idx0 : integer
        Index of the first image in the folder.

    """
    
    file_list = os.listdir(folder) # get the files in the directory
    img_amount = len(file_list)-5 # take away the 5 lvm files to get the amount
    frame0 = file_list[5] # get the name of the first image
    indices = [i for i, a in enumerate(frame0) if a == '.'] # find the '.' in the image name
    idx0 = int(frame0[indices[0]+1:indices[1]]) # extract the first index from the image name
    name_sliced = frame0[:(indices[0]+1)] # extract the nomenclature from the image name
    img_name = Fol_In + frame0 # directory of the image
    img = cv2.imread(img_name,0) # load the image
    # get the rotation angle and rotate
    angle = get_angle(img)
    rotated = ndimage.rotate(img, angle*180/np.pi) # rotate the image
    hr, wr = rotated.shape # get the new shape
    
    crop_rot =(0+int(np.rint(wr*np.tan(angle))), hr-int(np.rint(wr*np.tan(angle))),\
               0+int(np.rint(hr*np.tan(angle))), wr-int(np.rint(hr*np.tan(angle))))
        # crop coordinates to get rid of the black spots due to rotation
    cropped = rotated[crop_rot[0]:crop_rot[1],crop_rot[2]:crop_rot[3]] # crop the image
    
    new_peak_left, new_peak_right = find_edges(cropped) # get the new edges
    x_low =  get_most_common_element(new_peak_left) # get lower x bound
    x_high = get_most_common_element(new_peak_right) # get upper x bound
    crop_final = (0,cropped.shape[0],x_low+15,x_high+1-30) # prepare the final crop
    return crop_final, crop_rot, angle, name_sliced, img_amount, idx0

    
# set up the input and output folder
Fol_In = 'C:\PIV_Campaign\Fall\h2\F_h2_f1000_1_q'+os.sep
Fol_Out = 'C:\PIV_Processed\Images_Rotated\F_h2_f1000_1_q'+os.sep

# create the folder in case it doesn't exist
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

# get the parameters from the images in the folder
crop_final, crop_rot, angle, name_sliced, img_amount, idx0 = get_process_params(Fol_In)
# img_amount = 10

# iterate over all the images
for i in range(idx0, img_amount+idx0):
    img_name = name_sliced + '%06d.tif' %i # get the image name
    img = cv2.imread(Fol_In + img_name,0) # read the image
    img_processed = rotate_and_crop(img, crop_final, crop_rot, angle) # rotate and crop the image
    name_out = name_sliced + '%06d.tif' %i # crop the name for the output
    cv2.imwrite(Fol_Out + name_out ,img_processed) # write the cropped images to the output folder
    MEX = 'Cropping Image ' + str(i+1-idx0) + ' of ' + str(img_amount) # update the user
    print(MEX)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    