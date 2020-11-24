"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Code to animate velocity fields with quivers and contours
"""

import post_processing_functions as ppf # for reshaping the arrays
import matplotlib.pyplot as plt         # for plotting
import imageio                          # for rendering the gif
import numpy as np                      # for array operations
import os                               # for file paths
import shutil                           # for removing folders

# set the plot parameters in case they were changed in rc.params
ppf.set_plot_parameters()

# set the input folder
Fol_In = 'C:\\Users\manue\Desktop\data_files_16_64'
# set the folder in which to store gif images (will be deleted in the end)
Gif_Images = 'C:\\Users\manue\Desktop\Gif_Images'
if not os.path.exists(Gif_Images):
    os.makedirs(Gif_Images)

# get the amount of columns in the files
nx = ppf.get_column_amount(Fol_In)

# set the gif name and an empty list for the images to append into
GIFNAME = Gif_Images + os.sep + '..' +os.sep + 'rise_inversion.gif'
IMAGES = []
# give the starting frame and the amount of images to process
frame0 = 279
n_t = 40
# plot every IMG_STP_SZ'th image
IMG_STP_SZ =15

# set a custom colormap
custom_map = ppf.custom_div_cmap(100, mincol='indigo', midcol='darkcyan' ,maxcol='yellow')

# loop over the images
for i in range(0, n_t):
    # update user
    print('Image %d of %d' %((i+1, n_t)))
    # calculate the loadindex (we need this multiple times)
    load_idx = frame0+IMG_STP_SZ*i
    # load the data into arrays
    x, y, u, v = ppf.load_txt(Fol_In, load_idx, nx)
    # create the figure
    fig, ax = plt.subplots(figsize=(4.2,10))
    cs = plt.pcolormesh(x,y,v, vmin=-10, vmax=16, cmap = custom_map) # create the contourplot using pcolormesh
    ax.set_aspect('equal') # set the correct aspect ratio
    clb = fig.colorbar(cs) # get the colorbar
    clb.set_ticks(np.arange(-10, 17, 2)) # set the colorbarticks
    clb.ax.set_title('Velocity\n [px/frame] \n \n') # set the colorbar title
    # set a stepsize in the x and y direction to avoid overcrowding the quiver plot
    STEPY= 2
    STEPX = 1
    # plot the quivers
    ax.quiver(x[::STEPY, ::STEPX], y[::STEPY, ::STEPX], u[::STEPY, ::STEPX], v[::STEPY, ::STEPX],\
            color='k', scale =30, width=0.005,headwidth=4, headaxislength = 6)
    ax.set_title('Image %06d' % load_idx) # set the title
    Name_Out = Gif_Images + os.sep + '%06d.jpg' % (load_idx) # set the output name
    fig.savefig(Name_Out, dpi = 55) # save the figure
    IMAGES.append(imageio.imread(Name_Out)) # append into the imagelist
    plt.close(fig) # close the figure to avoid overcrowding the plot window
# render the gif
imageio.mimsave(GIFNAME, IMAGES, duration = 0.1)

# delete the image folder
shutil.rmtree(Gif_Images)