# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:56:58 2020

@author: Manuel Ratz
@description: Code to test the highpass filter to visualize the vorticies.
    This is just an example for one fall, the rest will be implemented in the 
    automatized postprocessing later
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np                       # for array operations
import post_processing_functions as ppf  # for loading of data, etc
import os                                # for data paths
import cv2                               # for reading of the raw tifs
import matplotlib.pyplot as plt          # for plotting the result
from scipy import ndimage                # for the high pass filtering
import imageio                           # for the gif rendering
import shutil                            # for deleting the folder in the end
#%%
# set up the folders containing the results and the raw images
Fol_Raw = 'C:\PIV_Processed\Images_Preprocessed\F_h2_f1000_1_q'+ os.sep
Fol_In = 'C:\PIV_Processed\Images_Processed\Results_F_h2_f1000_1_q_24_24' 

# load the data
NX = ppf.get_column_amount(Fol_In) # amount of columns
Im_Height, Im_Width = ppf.get_img_shape(Fol_Raw) # height and width of the raw image
Fol_Img = ppf.create_folder('Gif_Images') # folder for the gif images

Images = [] # empty list to append into
Gif_Name = 'Vorticy_fall.gif' # name of the final gif

# loop parameters
Frame0 = 310 # first frame with the vorticies
STP = 1 # stp size in time
N_T = 40 # number of images

# set up the loop
for i in range(0, N_T):
    Index = Frame0 + i*STP # loading index of the image
    print('Ímage %d of %d' %((i+1), N_T//STP)) # update the user
    # read the image and flip it vertically, this is to fit in with the calculated fields
    image_name = Fol_Raw + 'F_h2_f1000_1_q.%06d.tif' %Index 
    img = cv2.imread(image_name,0)
    img = np.flip(img, axis = 0)
    # load the data
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_In, Index, NX)
    #%%
    """
    Here we start with the filtering of the velocity fields to look for vorticies
    """
    # sigma of the filter and the truncation
    sigma = 6
    truncate = 5
    # get the blurred velocity field
    u_blur = ndimage.gaussian_filter(u, sigma = sigma, mode = 'nearest', truncate = truncate)
    v_blur = ndimage.gaussian_filter(v, sigma = sigma, mode = 'nearest', truncate = truncate)
    # subtract to get the high pass filtered velocity field
    u_filt = u - u_blur
    v_filt = v - v_blur
    
    
    #%%
    """
    This is cosmetics to make the plot look good and finally plot the result
    """
    
    # give the steps in case we want to skip rows and columns
    Stpx = 1
    Stpy = 1
    x = x[::Stpy, ::Stpx]
    y = y[::Stpy, ::Stpx]
    u = u[::Stpy, ::Stpx]
    v = v[::Stpy, ::Stpx]
    ratio = ratio[::Stpy, ::Stpx]
    mask = mask[::Stpy, ::Stpx]
    # valid and invalid points for the colors of the quiverplot
    invalid = mask.astype('bool')
    valid = ~invalid # mask for the colors
    
    # create the figure
    fig, ax = plt.subplots(figsize = (4, 8))
    ax.imshow(img, cmap = plt.cm.gray, alpha = 1) # show the iamge
    ax.axis('off') # disable the axis
    ax.invert_yaxis() # because image is inverted
    # scaling parameters for the plots
    Scale = 50 # for the arrow length
    Width = 0.004 # for the width of the base
    Headwidth = 4.5 # for the width of the head
    Headaxislength = 2.5 # for the length of the head
    ax.set_aspect(1) # equal aspect ratio in case we dont have the raw image
    # quiver plots, valid and invalid
    ax.quiver(x[valid], y[valid], u_filt[valid], v_filt[valid], color = 'lime', scale = Scale,\
              width = Width, headwidth = Headwidth, headaxislength = Headaxislength)
    ax.quiver(x[invalid], y[invalid], u_filt[invalid], v_filt[invalid], color = 'orangered', scale = Scale,\
              width = Width, headwidth = Headwidth, headaxislength = Headaxislength)
    ax.set_xlim(0, Im_Width) # in case the image is not plotted
    ax.set_ylim(Im_Height-800, Im_Height) # to better see the part close to the surface
    Name_Out = Fol_Img + os.sep + 'Vorticies_fall%06d.png' %Index # output name of the image
    fig.savefig(Name_Out, dpi = 150) # save the result
    Images.append(imageio.imread(Name_Out)) # append into the list
    plt.close(fig) # close in the plot window
imageio.mimsave(Gif_Name, Images, duration = 0.1) # render the gif

shutil.rmtree(Fol_Img) # delete the folder with the gif images