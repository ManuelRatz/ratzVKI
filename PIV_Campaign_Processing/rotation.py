#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:52:25 2020

@author: ratz
"""

import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

def rotate_image(image, wallcut = 0):
    """
    Description:
        This function takes the raw PIV images, calculates the angle they
        need to be rotated by and rotates them. Afterwards the edges are cut
        to have a rectangular channel again. The image is then cropped in the 
        width to only show the channel

    Parameters
    ----------
    image : 2d numpy array
        2d array of the image in greyscale.
    
    wallcut : Integer, optional
        Pixels to cut near the wall to filter out the plates. The default is 0.
    Returns
    -------
    angle
        Calculated angle of tilt for the image.

    """
    peak_left, peak_right= find_edges(image)
    
    # for the angle we calculate the median of 7 points at the edges to filter outliers
    # then we use the tan to calculate the rotation in radians
    angle_left = np.abs(np.arctan((np.median(peak_left[-7:])-np.median(peak_left[:7]))/img.shape[0]))
    angle_right = np.abs(np.arctan((np.median(peak_right[-7:])-np.median(peak_right[:7]))/img.shape[0]))
    
    # filter in case the angles differ by 1 percent
    # if(np.abs((angle_left-angle_right)/angle_left) > 0.01):
    #     print("Angle calculation failed")
    #     return np.nan
    
    # rotate the image and get the new height and width
    rotated = ndimage.rotate(image, angle_left*180/np.pi)
    hr, wr = rotated.shape
    
    # crop the rotated image to get rid of the edges
    crop_rot =(0+int(np.rint(wr*np.tan(angle_left))), hr-int(np.rint(wr*np.tan(angle_left))),\
           0+int(np.rint(hr*np.tan(angle_left))), wr-int(np.rint(hr*np.tan(angle_left))))
    cropped = rotated[crop_rot[0]:crop_rot[1],crop_rot[2]:crop_rot[3]]
    
    # find the new coordinates of the bright lines
    new_peak_left, new_peak_right = find_edges(cropped)
    """
    Get the new crop coordinates, for that we take the median of the new
    intensity indices, the +1 is because the cropping is exclusive.
    We also introduce a wallcut to get rid of the bright plates
    """
    wallcut = 7 # pixels to cut near the walls
    x_low =  get_most_common_element(new_peak_left)
    x_high = get_most_common_element(new_peak_right)
    crop_fin = (0,cropped.shape[0],x_low+wallcut,x_high+1-wallcut-10)
    final_img = cropped[crop_fin[0]:crop_fin[1],crop_fin[2]:crop_fin[3]]
    return final_img

def find_edges(image):
    """
    Description:
        Takes a 2d image of the channel and gets the edges of it.

    Parameters
    ----------
    image : 2d array containing the image as greyscale
        DESCRIPTION.

    Returns
    -------
    peak_left : 1d array
        Index of the highest intensity on the left side.
    peak_right : 1d array
        Index of the highest intensity on the right side.

    """
    grad = np.gradient(image, axis = 1) # take the gradient of the rows
    peak_left = np.argmax(grad>45, axis = 1) # find the first peak from the left side
    peak_right = img.shape[1]-np.argmax(grad[:,::-1]>25, axis =1) # find the first peak from the right side
    return peak_left, peak_right

def get_most_common_element(array):
    """
    Description: Get the most common element in an array

    Parameters
    ----------
    array : 1d Numpy array
        Array of interest.

    Returns
    -------
    idx : Integer
        Index of the most common value.

    """
    
    counts = np.bincount(array)
    idx = np.argmax(counts)    
    return idx

img = cv2.imread('R_h3_f1200_1_p15.6dcq31of.000773.tif',0) # load the image
rot = rotate_image(img)

# show the cropped rotation
fig, ax = plt.subplots()
plt.imshow(rot, cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('test.jpg',dpi = 800)
