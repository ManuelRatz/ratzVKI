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
Fol_In_1 = 'C:\PIV_Processed\Images_Processed\Results_Run_1_R_h1_f1200_1_p15'
Fol_In_2 = 'C:\PIV_Processed\Images_Processed\Results_Run_2_R_h1_f1200_1_p15'
Fol_In_3 = 'C:\PIV_Processed\Images_Processed\Results_Run_3_R_h1_f1200_1_p15'
Fol_In_4 = 'C:\PIV_Processed\Images_Processed\Results_Run_4_R_h1_f1200_1_p15'
Fol_In_5 = 'C:\PIV_Processed\Images_Processed\Results_Run_5_R_h1_f1200_1_p15'
Fol_In_6 = 'C:\PIV_Processed\Images_Processed\Results_Run_6_R_h1_f1200_1_p15'
Fol_In_7 = 'C:\PIV_Processed\Images_Processed\Results_Run_7_R_h1_f1200_1_p15'


# get the number of columns for each file
nx1 = ppf.get_column_amount(Fol_In_1)
nx2 = ppf.get_column_amount(Fol_In_2)
nx3 = ppf.get_column_amount(Fol_In_3)
nx4 = ppf.get_column_amount(Fol_In_4)
nx5 = ppf.get_column_amount(Fol_In_5)
nx6 = ppf.get_column_amount(Fol_In_6)
nx7 = ppf.get_column_amount(Fol_In_7)

# set up folders to store the gifs in (to be deleted later)
Gif_Fol_ver = 'gif_images_ver'
if not os.path.exists(Gif_Fol_ver):
    os.makedirs(Gif_Fol_ver)
Gif_Fol_hor = 'mp_gif_images_hor'
if not os.path.exists(Gif_Fol_hor):
    os.makedirs(Gif_Fol_hor)


# set the names of the gifs and empty lists to append into
name_hor = 'comparison_images' + os.sep + 'window_comparison_horizontal.gif'
name_ver = 'comparison_images' + os.sep + 'window_comparison_vertical.gif'
name_cor_hor = 'comparison_images' + os.sep + 'corr_comparison_horizontal.gif'
name_cor_ver = 'comparison_images' + os.sep + 'corr_comparison_vertical.gif'
images_hor = []
images_ver = []
images_cor_hor = []
images_cor_ver = []

# set the indices for the horizontal and vertical direction
# these are the indices at which the interrogation windows have the same coordinates
IDY1 = 3
IDY2 = 5
IDY3 = 7
IDY4 = 5
IDY5 = 5
IDY6 = 2

IDX1 = 1
IDX2 = 2
IDX3 = 3
IDX4 = 2
IDX5 = 6
IDX6 = 6

# set the number of imagess and the frame at which to start
n_t = 15
frame0 = 279

for k in range(0, n_t):
    # set the loading index (we need this multiple times)
    load_index = 2*k+frame0
    # update the user
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    # load the data into the arrays
    x1, y1, u1, v1 = ppf.load_txt(Fol_In_1, load_index, nx1)
    x2, y2, u2, v2 = ppf.load_txt(Fol_In_2, load_index, nx2)
    x3, y3, u3, v3 = ppf.load_txt(Fol_In_3, load_index, nx3)
    x4, y4, u4, v4 = ppf.load_txt(Fol_In_4, load_index, nx4)
    x5, y5, u5, v5 = ppf.load_txt(Fol_In_5, load_index, nx5)
    x6, y6, u6, v6 = ppf.load_txt(Fol_In_6, load_index, nx6)
    x7, y7, u7, v7 = ppf.load_txt(Fol_In_7, load_index, nx7)
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    # plot the 4 data sets with scatter plots and lines
    ax.plot(x1[IDX1,:],v1[IDX1,:], label = '1', c='b')
    ax.scatter(x1[IDX1,:],v1[IDX1,:], marker='x', s=(300./fig.dpi)**2, c='b')
    ax.plot(x2[IDX2,:],v2[IDX2,:], label = '2', c='r')
    ax.scatter(x2[IDX2,:],v2[IDX2,:], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.plot(x3[IDX3,:],v3[IDX3,:], label = '3', c='y')
    ax.scatter(x3[IDX3,:],v3[IDX3,:], marker='x', s=(300./fig.dpi)**2, c='y')
    ax.plot(x4[IDX4,:],v4[IDX4,:], label = '4', c='g')
    ax.scatter(x4[IDX4,:],v4[IDX4,:], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.plot(x5[IDX5,:],v5[IDX5,:], label = '5', c='c')
    ax.scatter(x5[IDX5,:],v5[IDX5,:], marker='x', s=(300./fig.dpi)**2, c='c')
    ax.plot(x6[IDX6,:],v6[IDX6,:], label = '6', c='m')
    ax.scatter(x6[IDX6,:],v6[IDX6,:], marker='x', s=(300./fig.dpi)**2, c='m')
    ax.grid(b = True, lw = 1) # enable the grid
    ax.legend(loc = 'lower center', ncol = 2) # set the legend
    ax.set_ylabel('v[px/frame]') # label the y axis
    ax.set_xlabel('$x$[px]') # label the x axis
    ax.set_ylim(0,15) # set the y limit
    ax.set_xlim(0,270) # set the x limit
    plt.title('Frame %03d' %(load_index)) # set the image title
    save_name = Gif_Fol_ver + os.sep + 'horizontal_refinement_%06d.jpg' %load_index # set the image name
    fig.savefig(save_name, dpi = 75) # save the figure
    images_hor.append(imageio.imread(save_name)) # append into the imagelist
    plt.close(fig)
    
      # repeat the same thing for the vertical plots
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(y1[:,IDY1],v1[:,IDY1], label = '1')
    ax.scatter(y1[:,IDY1],v1[:,IDY1], marker='x', s=(300./fig.dpi)**2)
    ax.plot(y2[:,IDY2],v2[:,IDY2], label = '2', c='r')
    ax.scatter(y2[:,IDY2],v2[:,IDY2], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.plot(y3[:,IDY3],v3[:,IDY3], label = '3', c='y')
    ax.scatter(y3[:,IDY3],v3[:,IDY3], marker='x', s=(300./fig.dpi)**2, c='y')
    ax.plot(y4[:,IDY4],v4[:,IDY4], label = '4', c='g')
    ax.scatter(y4[:,IDY4],v4[:,IDY4], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.plot(y5[:,IDY5],v5[:,IDY5], label = '5', c='c')
    ax.scatter(y5[:,IDY5],v5[:,IDY5], marker='x', s=(300./fig.dpi)**2, c='c')
    ax.plot(y6[:,IDY6],v6[:,IDY6], label = '6', c='m')
    ax.scatter(y6[:,IDY6],v6[:,IDY6], marker='x', s=(300./fig.dpi)**2, c='m')
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 6)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(12,16)
    ax.set_xlim(0,1230)
    plt.title('Frame %03d' %(load_index))
    save_name = Gif_Fol_hor + os.sep + 'vertical_refinement.jpg'
    fig.savefig(save_name, dpi = 75)
    images_ver.append(imageio.imread(save_name))
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(y3[:,IDY3],v3[:,IDY3], label = 'Circular', c='g')
    ax.scatter(y3[:,IDY3],v3[:,IDY3], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.plot(y7[:,IDY3],v7[:,IDY3], label = 'Linear', c='r')
    ax.scatter(y7[:,IDY3],v7[:,IDY3], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 2)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(12,16)
    ax.set_xlim(0,1230)
    plt.title('Frame %03d' %(load_index))
    save_name = Gif_Fol_ver + os.sep + 'vertical_corr.jpg'
    fig.savefig(save_name, dpi = 75)
    images_cor_ver.append(imageio.imread(save_name))
    plt.close(fig) 
    
    fig, ax = plt.subplots(figsize = (8,5))    
    ax.plot(x3[IDX3,:],v3[IDX3,:], label = 'Circular', c='g')
    ax.scatter(x3[IDX3,:],v3[IDX3,:], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.plot(x7[IDX3,:],v7[IDX3,:], label = 'Circular', c='r')
    ax.scatter(x7[IDX3,:],v7[IDX3,:], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 2)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(0,15)
    ax.set_xlim(0,270)
    plt.title('Frame %03d' %(load_index))
    save_name = Gif_Fol_hor + os.sep + 'horizontal_corr.jpg'
    fig.savefig(save_name, dpi = 75)
    images_cor_hor.append(imageio.imread(save_name))
    plt.close(fig)
# render the gifs
imageio.mimsave(name_hor, images_hor, duration = 1.25)
imageio.mimsave(name_ver, images_ver, duration = 1.25)
imageio.mimsave(name_cor_hor, images_cor_hor, duration = 0.75)
imageio.mimsave(name_cor_ver, images_cor_ver, duration = 0.75)

# delete the folder containing the temporary plot images
import shutil
shutil.rmtree(Gif_Fol_hor)
shutil.rmtree(Gif_Fol_ver)    
    
    
    
    
    
    
    
    
    
    
    
    
    