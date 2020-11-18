"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Code to compare different refinements for the processing of a rise
"""

import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for reshaping the arrays
import os
import imageio

# set the default plot parameters
ppf.set_plot_parameters()

# set input and output folders
Fol_In_1 = 'C:\\Users\manue\Desktop\\text_files_24_96'
Fol_In_2 = 'C:\\Users\manue\Desktop\\text_files_12_48'
Fol_In_3 = 'C:\\Users\manue\Desktop\\text_files_16_64'
Fol_In_4 = 'C:\\Users\manue\Desktop\\text_files_16_64_Run_2'
# Fol_In_2 = 'C:\PIV_Processed\Images_Processed\Results_16_64_R_h1_f1200_1_p15\\text_files'

# get the number of columns for each file
nx1 = ppf.get_column_amount(Fol_In_1)
nx2 = ppf.get_column_amount(Fol_In_2)
nx3 = ppf.get_column_amount(Fol_In_3)
nx4 = ppf.get_column_amount(Fol_In_4)

# set up folders to store the gifs in (to be deleted later)
Gif_Fol_ver = 'tmp_gif_images'+os.sep
if not os.path.exists(Gif_Fol_ver):
    os.makedirs(Gif_Fol_ver)
Gif_Fol_hor = 'tmp_gif_images'+os.sep
if not os.path.exists(Gif_Fol_hor):
    os.makedirs(Gif_Fol_hor)

# set the names of the gifs and empty lists to append into
name_hor = 'comparison_images' + os.sep + 'window_comparison_horizontal.gif'
name_ver = 'comparison_images' + os.sep + 'window_comparison_vertical.gif'
images_hor = []
images_ver = []

# set the indices for the horizontal and vertical direction
# these are the indices at which the interrogation windows have the same coordinates
IDX1 = 3
IDX2 = 7
IDX3 = 5
IDX4 = 5

IDY1 = 1
IDY2 = 3
IDY3 = 2
IDY4 = 2

# set the number of imagess and the frame at which to start
n_t = 1
frame0 = 309

for k in range(0, n_t):
    # set the loading index (we need this multiple times)
    load_index = 3*k+frame0
    # update the user
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    # load the data into the arrays
    x1, y1, u1, v1 = ppf.load_txt(Fol_In_1, load_index, nx1)
    x2, y2, u2, v2 = ppf.load_txt(Fol_In_2, load_index, nx2)
    x3, y3, u3, v3 = ppf.load_txt(Fol_In_3, load_index, nx3)
    x4, y4, u4, v4 = ppf.load_txt(Fol_In_4, load_index, nx4)
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    # plot the 4 data sets with scatter plots and lines
    ax.plot(x1[IDX1,:],v1[IDX1,:], label = 'linear')
    ax.scatter(x1[IDX1,:],v1[IDX1,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x2[IDX2,:],v2[IDX2,:], label = 'circular', c='r')
    ax.scatter(x2[IDX2,:],v2[IDX2,:], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.plot(x3[IDX3,:],v3[IDX3,:], label = '16 64 Rough Start', c='y')
    ax.scatter(x3[IDX3,:],v3[IDX3,:], marker='x', s=(300./fig.dpi)**2, c='y')
    ax.plot(x4[IDX4,:],v4[IDX4,:], label = '16 64 Fine Start', c='g')
    ax.scatter(x4[IDX4,:],v4[IDX4,:], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.grid(b = True, lw = 1) # enable the grid
    ax.legend(loc = 'lower center', ncol = 2) # set the legend
    ax.set_ylabel('v[px/frame]') # label the y axis
    ax.set_xlabel('$x$[px]') # label the x axis
    ax.set_ylim(0,15) # set the y limit
    ax.set_xlim(0,270) # set the x limit
    plt.title('Frame %03d' %(load_index)) # set the image title
    save_name = Gif_Fol_ver + 'circ_vs_lin_horizontal_%06d.jpg' %load_index # set the image name
    fig.savefig(save_name, dpi = 400) # save the figure
    images_hor.append(imageio.imread(save_name)) # append into the imagelist
    
     # repeat the same thing for the vertical plots
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(y1[:,IDY1],v1[:,IDY1], label = 'linear')
    ax.scatter(y1[:,IDY1],v1[:,IDY1], marker='x', s=(300./fig.dpi)**2)
    ax.plot(y2[:,IDY2],v2[:,IDY2], label = 'circular', c='r')
    ax.scatter(y2[:,IDY2],v2[:,IDY2], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.plot(y3[:,IDY3],v3[:,IDY3], label = '16 64 Rough Start', c='y')
    ax.scatter(y3[:,IDY3],v3[:,IDY3], marker='x', s=(300./fig.dpi)**2, c='y')
    ax.plot(y4[:,IDY4],v4[:,IDY4], label = '16 64 Fine Start', c='g')
    ax.scatter(y4[:,IDY4],v4[:,IDY4], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 2)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(14,16)
    ax.set_xlim(0,1230)
    plt.title('Frame %03d' %(load_index))
    save_name = Gif_Fol_hor + 'circ_vs_lin_vertical.jpg'
    fig.savefig(save_name, dpi = 400)
    images_ver.append(imageio.imread(save_name))

# render the gifs
imageio.mimsave(name_hor, images_hor, duration = 0.5)
imageio.mimsave(name_ver, images_ver, duration = 0.5)

# delete the folder containing the temporary plot images
import shutil
shutil.rmtree(Gif_Fol_hor)
shutil.rmtree(Gif_Fol_ver)    
    
    
    
    
    
    
    
    
    
    
    
    
    