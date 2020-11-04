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
import time                 # for calculating runtime

def get_angle(img_avg):
    """
    Function to calculate the angle of tilt for the average image

    Parameters
    ----------
    img_avg : 2d array of uint 8
        Average Grayscale image of the channel.

    Returns
    -------
    angle_left : float64
        Angle of tilt of the image.

    """
    def linear(a, b, x):
        """
        Help function for the linear fit of the images
        """
        return a*x+b
    # get a dummy for fitting purposes
    x = np.arange(1,img_avg.shape[0]+1,1)
    # get the left edge of the channel
    peak_left = np.argmax(img_avg[:,:400], axis = 1).astype(np.float64)
    # get the median for filtering
    median = np.median(peak_left)
    # filter out by replacing outliers with nans
    for i in range(0,len(peak_left)):
        if(np.abs(median-peak_left[i]) > 10):
            peak_left[i] = np.nan
    # filter out the nans
    idx = np.isfinite(peak_left)
    # do a linear fit of the detected edges
    a,b = np.polyfit(x[idx],peak_left[idx], 1) 
    # calculate the angle
    angle_left = np.abs(np.arctan((linear(a,b,1280)-b)/1280))
    return angle_left

def get_avg_image(folder, name_sliced, idx0):
    """
    Function to calculate an average of selected images

    Parameters
    ----------
    folder : string
        Data path to the images.
    name_sliced : string
        Beginning of the image name for loading.
    idx0 : int
        Index of the first image in the folder.

    Returns
    -------
    img_avg : 2d array of uint8
        Averaged image as a 2d array in grayscale.

    """
    # get the name of the first image in the folder
    img_name = folder + name_sliced + '.%06d.tif' %idx0
    # read the image
    img = cv2.imread(img_name,0)
    # get the dimensions
    ny, nx = img.shape
    # set the amount of images to calculate
    n_t = 50
    # initialize the data matrix
    Data = np.zeros((nx * ny, n_t))
    # check whether we have a Rise or a Fall. For a Fall take the first 50 images
    # for a Rise we take the images from 470 to 520 as these contain particles
    # for all the test cases
    if(name_sliced[0] == 'F'):
        shift = idx0
    else:
        shift = 470
    # iterate over the images
    for i in range(0,n_t):
        # set the input name
        img_name = folder + name_sliced+ '.%06d.tif' %(i+shift)
        # load the image
        img = cv2.imread(img_name,0)
        # reshape into a column vector
        img_column =  np.reshape(img, ((nx * ny, 1)))
        # save in the data matrix
        Data[:,i] = img_column[:,0]
    # calculate the average image
    avg = np.mean(Data, axis = 1)
    # reshape it into a 2d array
    img_avg = np.reshape(avg, ((ny, nx)))
    return img_avg

def find_edges(image):
    """
    Function to find the edges for the cropping.
    This function only works if the channel is somewhat centered as we are 
    searching for the peaks left and right of the center. If this is not the case
    just change the 'half_idx' variable

    Parameters
    ----------
    image : 2d array of uint8
        Grayscale image of the channel.

    Returns
    -------
    peak_left : 1d array of int32
        Indices of the left edge.
    peak_right : 1d array of int32
        Indices of the right edge.

    """
    # get the index of the half of the image
    half_idx = int(image.shape[1]/2)
    # find the peak from the left side
    peak_left = np.argmax(image[:,:half_idx], axis = 1).astype(np.float64)
    # get the left median for filtering
    median_left = np.median(peak_left)
    # filter the outliers by setting nans
    for i in range(0,len(peak_left)):
        if(np.abs(median_left-peak_left[i]) > 10):
            peak_left[i] = np.nan
    # filter out the nans for later
    idx_l = np.isfinite(peak_left)
    peak_left = peak_left[idx_l]
    # find the peak from the right side
    peak_right = half_idx+np.argmax(image[:,half_idx:], axis =1).astype(np.float64) # find the first peak from the right side
    # get the right median for filtering
    median_right = np.median(peak_right)
    # filter out outliers by setting nans
    for i in range(0,len(peak_right)):
        if(np.abs(median_right-peak_right[i]) > 10):
            peak_right[i] = np.nan
    # filter out the nans for later
    idx_r = np.isfinite(peak_right)
    peak_right = peak_right[idx_r]
    return peak_left, peak_right

def get_most_common_element(array):
    """
    Function to find the most common element in a numpy array

    Parameters
    ----------
    array : 1d array of uint8
        Array of interest.

    Returns
    -------
    idx : integer
        Most common element in the array. In our case this is an index.

    """
    # get the histogram of the array
    counts = np.bincount(array.astype(np.int))
    # get the peak
    idx = np.argmax(counts)    
    return idx

def rotate_and_crop(image, crop_final, crop_rot, angle):
    """
    Rotate and crop an image

    Parameters
    ----------
    image : 2d array of uint8
        Grayscale image of the channel.
    crop_final : list of 4 int32
        Contains the final crop used to get just the channel in the image.
    crop_rot : list of 4 int32
        Contains the Crop of the rotated image to cut the edges.
    angle : float64
        Angle of tilt for the image.

    Returns
    -------
    final_img : 2d array of uint8
        Final image for PIV processing, cropped and rotated accordingly.

    """
    # rotate the image
    rotated = ndimage.rotate(image, angle*180/np.pi)
    # crop the corners after the rotation
    cropped = rotated[crop_rot[0]:crop_rot[1],crop_rot[2]:crop_rot[3]]
    # crop to the final width
    final_img = cropped[crop_final[0]:crop_final[1],crop_final[2]:crop_final[3]]
    return final_img

def get_process_params(folder, wallcut_file):
   
    """
    Function to get the rotation angle and crop range

    Parameters
    ----------
    folder : string
        Input folder where the images are located.

    Returns
    -------
    crop_final : 2d array of int32
        Array containing the 4 crop parameters at the end.
    crop_rot : 2d array of int32
        Array containing the 4 crop parameters after the rotation.
    angle : float64
        Angle of tilt for the images in radians.
    name_sliced : string
        Name of the images without the number.
    img_amount : int
        Amount of images in the folder.
    idx0 : int
        Index of the first image in the folder.
    images_exist : boolean
        True if there are images in the folder, otherwise false

    """
    run_names = wallcut_file[:,0].astype(np.str)
    wallcuts = wallcut_file[:,1:].astype(np.float)
    # get a list of all the files in the folder
    file_list = os.listdir(folder)
    # get the amount of images by taking away the labview files
    img_amount = len(file_list)-5
    # check wether there actually are images in the folder
    if(img_amount < 1):
        # print error message and return
        MEX = 'No images in folder %s' %folder
        print(MEX)
        return 0, 0, 0, 0, 0, 0, False
    # get the name of the first frame
    frame0 = file_list[5]
    # get the '.' in the file name
    indices = [i for i, a in enumerate(frame0) if a == '.']
    # extract the index of the first frame
    idx0 = int(frame0[indices[0]+1:indices[1]])
    # cut of the index of the file to get the sliced name
    name_sliced = frame0[:(indices[0])]
    # search for the right wallcut
    for j in range(0, len(run_names)):
        if(name_sliced == run_names[j]):
            wallcut_left, wallcut_right = wallcuts[j,:]
    # check whether a wallcut was found
    if (wallcut_left ==np.nan):
        MEX = 'No wallcut was found'
        print(MEX)
        return 0, 0, 0, 0, 0, 0, False
    # calculate the average image
    avg = get_avg_image(folder, name_sliced, idx0)
    # calculate the angle from the average image
    angle = get_angle(avg)
    # rotate the image
    rotated = ndimage.rotate(avg, angle*180/np.pi)
    # get the new image shape
    hr, wr = rotated.shape
    # crop the edges due to the rotation
    crop_rot =(0+int(np.rint(wr*np.tan(angle))), hr-int(np.rint(wr*np.tan(angle))),\
               0+int(np.rint(hr*np.tan(angle))), wr-int(np.rint(hr*np.tan(angle))))
    cropped = rotated[crop_rot[0]:crop_rot[1],crop_rot[2]:crop_rot[3]]
    # get the edges of the rotated image
    new_peak_left, new_peak_right = find_edges(cropped) 
    x_low =  get_most_common_element(new_peak_left)
    x_high = get_most_common_element(new_peak_right)
    # arange coordinates of the final crop
    crop_final = (0,cropped.shape[0],x_low+int(wallcut_left),x_high+1-int(wallcut_right)) 
    return crop_final, crop_rot, angle, name_sliced, img_amount, idx0, True

  
"""
This part does the rotation for a single run, one can set the folder and how
many images to process
"""
   
# # set up the input and output folder  
# Fol_In = 'C:\PIV_Campaign\Rise\h4\p1500\R_h4_f1200_1_p15' + os.sep
# Fol_Out = 'C:\PIV_Processed\Images_Rotated\R_h4_f1200_1_p15' + os.sep
#
# wallcut_file = np.genfromtxt('wallcuts.txt', dtype = str)
# crop_final, crop_rot, angle, name_sliced, img_amount, idx0, images_exist =\
#             get_process_params(Fol_In, wallcut_file)
# if(images_exist):
#     Fol_Out = 'C:\PIV_Processed\Images_Rotated'+os.sep+name_sliced+os.sep
#     # create the folder in case it doesn't exist
#     if not os.path.exists(Fol_Out):
#         os.mkdir(Fol_Out)
#     img_amount = 10
#     # iterate over all the images
#     for i in range(idx0+470, img_amount+idx0+470):
#         img_name = name_sliced + '.%06d.tif' %i # get the image name
#         img = cv2.imread(Fol_In + img_name,0) # read the image
#         img_processed = rotate_and_crop(img, crop_final, crop_rot, angle) # rotate and crop the image
#         name_out = name_sliced + '.%06d.tif' %i # crop the name for the output
#         cv2.imwrite(Fol_Out + name_out ,img_processed) # write the cropped images to the output folder
#         MEX = 'Cropping Image ' + str(i+1-idx0) + ' of ' + str(img_amount)\
#             + ' for run %s' %name_sliced# update the user
#         print(MEX)
# else:
#     MEX = 'No images found in directory %s' %Fol_In
#     print(MEX)
    
"""
This part calculates the rotated images for ALL the files
WARNING: This is a very long code ~5 hours as we are processing 180 GB of data
"""

###############################################################################
###                      First we calculate the Falls                       ###
###############################################################################

# folder containing all the falls
Fol_In = 'C:'+os.sep+'PIV_Campaign'+os.sep+'Fall'+os.sep
# load the wallcut file
wallcut_file = np.genfromtxt('wallcuts.txt', dtype = str)


# # get all of the heights and loop over them
# heights = os.listdir(Fol_In)
# for m in range(1,4):
#     # get all the speeds and loop over them
#     speeds = os.listdir(Fol_In+heights[m])
#     for j in range(0,len(speeds)):
#         # get the start time of the run
#         start = time.process_time()
#         # set the current working directory
#         current_folder = Fol_In+heights[m]+os.sep+speeds[j]+os.sep
#         # get the parameters from the images in the folder
#         crop_final, crop_rot, angle, name_sliced, img_amount, idx0, images_exist =\
#             get_process_params(current_folder, wallcut_file)
#         # check whether there are images
#         if(images_exist):
#             # create the folder in case it doesn't exist
#             Fol_Out = 'C:\PIV_Processed\Images_Rotated'+os.sep+name_sliced+os.sep
#             if not os.path.exists(Fol_Out):
#                 os.mkdir(Fol_Out)
#             # optionally set the image amount here
#             # img_amount = 10
#             # iterate over all the images
#             for i in range(idx0, img_amount+idx0):
#                 img_name = name_sliced + '.%06d.tif' %i
#                 img = cv2.imread(current_folder + img_name,0)
#                 img_processed = rotate_and_crop(img, crop_final, crop_rot, angle)
#                 name_out = name_sliced + '.%06d.tif' %i
#                 cv2.imwrite(Fol_Out + name_out ,img_processed)
#                 if (((i-idx0+1)%100) == 0):
#                     MEX = 'Cropping Image ' + str(i+1-idx0) + ' of ' + str(img_amount)\
#                         + ' for run %s' %name_sliced
#                     print(MEX)
#         else:
#             MEX = 'No images found in directory %s' %Fol_In
#             print(MEX)
#         # print the time
#         print(time.process_time() - start)   
  
    
###############################################################################
###                          And now the Rises                              ###
############################################################################### 

# folder containing all the rises
Fol_In = 'C:'+os.sep+'PIV_Campaign'+os.sep+'Rise'+os.sep
# load the wallcut file
wallcut_file = np.genfromtxt('wallcuts.txt', dtype = str)

# get all of the heights and loop over them
heights = os.listdir(Fol_In)
# for m in range(0,len(heights)):
for m in range(3, 4):
    # get all the pressures and loop over them
    pressures = os.listdir(Fol_In+heights[m])
    for k in range(0, len(pressures)):
    # for k in range(0, 3):
        # get all the different runs and loop over them
        runs = os.listdir(Fol_In+heights[m]+os.sep + pressures[k])
        for j in range(0, len(runs)):
            # get the start time of the run
            start = time.process_time()
            # set the current working directory
            current_folder = Fol_In+heights[m]+os.sep + pressures[k]+os.sep+runs[j]+os.sep
            # get the parameters from the images in the folder
            crop_final, crop_rot, angle, name_sliced, img_amount, idx0, images_exist =\
                get_process_params(current_folder, wallcut_file)
            # check whether there are images
            if(images_exist):
                # create the folder in case it doesn't exist
                Fol_Out = 'C:\PIV_Processed\Images_Rotated'+os.sep+name_sliced+os.sep
                if not os.path.exists(Fol_Out):
                    os.mkdir(Fol_Out)
                # optionally set the image amount here
                # img_amount = 3
                # iterate over all the images
                for i in range(idx0, img_amount+idx0):
                    # set te image name and load it
                    img_name = name_sliced + '.%06d.tif' %i
                    img = cv2.imread(current_folder + img_name,0)
                    # process the image
                    img_processed = rotate_and_crop(img, crop_final, crop_rot, angle)
                    # save the image
                    name_out = name_sliced + '.%06d.tif' %i
                    cv2.imwrite(Fol_Out + name_out ,img_processed)
                    # update the user every 100 images
                    if (((i-idx0+1)%100) == 0):
                        MEX = 'Cropping Image ' + str(i+1-idx0) + ' of ' + str(img_amount)\
                            + ' for run %s' %name_sliced
                        print(MEX)
            else:
                MEX = 'No images found in directory %s' %Fol_In
                print(MEX)
            # print the time
            print(time.process_time() - start)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    