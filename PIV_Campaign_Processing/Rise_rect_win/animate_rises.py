# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Code to animate one fully processed rise for the presentation 
    at the sloshing meeting
"""

import os                        # for file paths
import imageio                   # for rendering the gif
import matplotlib.pyplot as plt  # for plotting
import numpy as np               # for calculations and arrays
import post_processing_functions as ppf # for reshaping the arrays

ppf.set_plot_parameters()

# set input and output folders
Fol_In = 'C:\\Users\manue\Desktop\\tmp_processed\Open_PIV_results_16_64_R_h1_f1200_1_p14' + os.sep
# folder for the gifs with constant x
Fol_Gif_x = 'C:\\Users\manue\Desktop' + os.sep + 'Gif_images_x' + os.sep
if not os.path.exists(Fol_Gif_x):
    os.mkdir(Fol_Gif_x)
# folder for the gifs with constant y
Fol_Gif_y = 'C:\\Users\manue\Desktop' + os.sep + 'Gif_images_y' + os.sep
if not os.path.exists(Fol_Gif_y):
    os.mkdir(Fol_Gif_y)

# set up the names and empty lists to append the images into
GIFNAME_X = Fol_Gif_x + '..' + os.sep + 'rise_constant_x.gif' # name of the gif 
images_x = [] # empty list for the image names
GIFNAME_Y = Fol_Gif_y + '..' + os.sep + 'rise_constant_y.gif' # name of the gif 
images_y = [] # empty list for the image names
# give the index of the txt at which to start the calculation
nx = ppf.get_column_amount(Fol_In)
x, y, u, v = ppf.load_txt(Fol_In, 0, nx)
# number of steps (if we want to process everything)
n_t = len(os.listdir(Fol_In))  

IDX0 = 5  # first x index
IDX1 = 19 # middle x index
IDX2 = 33 # last x index
IDY0 = 4  # first y index
IDY1 = int(len(x[0,:])/2) # middle y index
IDY2 = int(len(x[0,:])-4) # last y index
n_t = 200 # custom number of images
frame0 = 30
for k in range(0, 0+n_t):
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    # give the name of the current image
    loadindex = 3*k+frame0
    NAME = Fol_In + os.sep + 'field_A%06d' % loadindex + '.txt' 
    x, y, u, v = ppf.load_txt(Fol_In, loadindex, nx)
    
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    # plot the three lines with scatter plots to see individual points
    ax.plot(x[IDX0,:],v[IDX0,:], label = 'y = %d px' %y[IDX0,0])
    ax.scatter(x[IDX0,:],v[IDX0,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x[IDX1,:],v[IDX1,:], label = 'y = %d px' %y[IDX1,0])
    ax.scatter(x[IDX1,:],v[IDX1,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x[IDX2,:],v[IDX2,:], label = 'y = %d px' %y[IDX2,0])
    ax.scatter(x[IDX2,:],v[IDX2,:], marker='x', s=(300./fig.dpi)**2)    
    
    ax.grid(b = True, lw = 2) # enable the grid
    ax.legend(loc = 'lower center', ncol = 3) # enable the legend
    ax.set_ylabel('v[px/frame]') # set y label
    ax.set_xlabel('$x$[px]') # set x label
    ax.set_ylim(-10,15) # set the y limit
    ax.set_xlim(x[0,0],x[0,-1]) # set the x limit
    plt.title('Frame %03d' %(loadindex)) # give the title to keep track
    Save_Name = Fol_Gif_y +'Gif_img%06d.png' %k # set the output name
    fig.savefig(Save_Name,dpi=60) # save the plot
    plt.close(fig) # close the figure to avoid overcrowding
    images_y.append(imageio.imread(Save_Name)) # append the name into the list of images
    
    # repeat the same thing for the plots along the vertical axis
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(y[:,IDY0], v[:,IDY0], label='x = %d px' %x[0,IDY0])
    ax.scatter(y[:,IDY0], v[:,IDY0], marker='x', s=(300./fig.dpi)**2)
    ax.plot(y[:,IDY1], v[:,IDY1], label = 'x = %d px' %x[0,IDY1])
    ax.scatter(y[:,IDY1], v[:,IDY1], marker='x', s=(300./fig.dpi)**2)
    ax.plot(y[:,IDY2], v[:,IDY2], label = 'x = %d px' %x[0,IDY2])
    ax.scatter(y[:,IDY2], v[:,IDY2], marker='x', s=(300./fig.dpi)**2)
    ax.legend(loc = 'lower center', ncol = 3)
    ax.grid(b = True, lw = 2)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$y$[px]')
    ax.set_ylim(-10,15)
    ax.set_xlim(y[-1,0],y[0,0])
    plt.title('Frame %03d' %(loadindex))
    Save_Name = Fol_Gif_x +'Gif_img%06d.png' %k
    fig.savefig(Save_Name,dpi=60)
    plt.close(fig)
    images_x.append(imageio.imread(Save_Name)) 
    
imageio.mimsave(GIFNAME_X, images_x, duration=0.1) # create the gif
imageio.mimsave(GIFNAME_Y, images_y, duration=0.1) # create the gif

# delete the folders where we assembled the gif images
import shutil
shutil.rmtree(Fol_Gif_x)
shutil.rmtree(Fol_Gif_y)






