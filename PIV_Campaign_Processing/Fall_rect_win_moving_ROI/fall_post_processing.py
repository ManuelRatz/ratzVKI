# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:40:18 2019

@author: mendez
"""

import os  # This is to understand which separator in the paths (/ or \)
import imageio # for rendering the gif
import matplotlib.pyplot as plt  # This is to plot things
import numpy as np  # This is for doing math

# define some plot parameters
plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)   # fontsize of the figure title
# plt.rc('text', usetex=True)      # use latex for the text
plt.rc('font', family='serif')   # serif as text font
# plt.rc('axes', grid=True)        # enable the grid
# plt.rc('savefig', dpi = 100)     # set the dpi for saving figures

run = 'F_h3_f1200_1_s_24_24'

# set input and output folders
Fol_In = 'D:\PIV_Processed\Images_Processed\Open_PIV_results_' + run + os.sep
# Fol_Out = 'D:\PIV_Processed\Images_Postprocessed'+os.sep
# if not os.path.exists(Fol_Out):
#     os.mkdir(Fol_Out)
Fol_Gif_x = 'D:\PIV_Processed\Images_Postprocessed' + os.sep+run + '_Gif_images_x' + os.sep
if not os.path.exists(Fol_Gif_x):
    os.mkdir(Fol_Gif_x)
Fol_Gif_y = 'D:\PIV_Processed\Images_Postprocessed' + os.sep+run + '_Gif_images_y' + os.sep
if not os.path.exists(Fol_Gif_y):
    os.mkdir(Fol_Gif_y)


GIFNAME_X = Fol_Gif_x + '..' + os.sep + run + '_constant_x.gif' # name of the gif 
images_x = [] # empty list for the image names
GIFNAME_Y = Fol_Gif_y + '..' + os.sep + run + '_constant_y.gif' # name of the gif 
images_y = [] # empty list for the image names
frame0 = 136
NAME = Fol_In + os.sep + 'field_A%06d' % frame0 + '.txt'  # Check it out: print(Name)
data = np.genfromtxt(NAME)  # Here we have the four colums
nxny = data.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y

X_S = data[:, 0]
Y_S = data[:, 1]
GRAD_Y = np.diff(Y_S)
IND_X = np.where(GRAD_Y != 0)
DAT = IND_X[0]
n_y = DAT[0] + 1
n_x = (nxny // (n_y))  # Carefull with integer and float!
x = (X_S.reshape((n_x, n_y)))
y = (Y_S.reshape((n_x, n_y)))
n_t = len(os.listdir(Fol_In))  # number of steps.

# d_1 = np.zeros((n_t, n_x))
IDX0 = 10
IDX1 = 50
IDX2 = 90
IDY0 = 4
IDY1 = int(len(x[0,:])/2)
IDY2 = int(len(x[0,:])-4)
n_t = 48
for k in range(0, n_t):
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    NAME = Fol_In + os.sep + 'field_A%06d' % (5*k+frame0) + '.txt' 
    DATA = np.genfromtxt(NAME)  # Here we have the four colums
    current_len = DATA.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    if(current_len != nxny):
        break
    x = (X_S.reshape((n_x, n_y)))
    y = (Y_S.reshape((n_x, n_y)))
    V_X = DATA[:, 2]  # U component
    V_Y = DATA[:, 3]
    Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
    u = (V_X.reshape((n_x, n_y)))
    v = (V_Y.reshape((n_x, n_y)))
    Magn = (Mod.reshape((n_x, n_y)))
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x[IDX0,:],v[IDX0,:], label = 'y = %d px' %y[IDX0,0])
    ax.scatter(x[IDX0,:],v[IDX0,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x[IDX1,:],v[IDX1,:], label = 'y = %d px' %y[IDX1,0])
    ax.scatter(x[IDX1,:],v[IDX1,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x[IDX2,:],v[IDX2,:], label = 'y = %d px' %y[IDX2,0])
    ax.scatter(x[IDX2,:],v[IDX2,:], marker='x', s=(300./fig.dpi)**2)    
    ax.grid(b = True, lw = 2)
    ax.legend(loc = 'lower center', ncol = 3)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(-14,0)
    ax.set_xlim(x[0,0],x[0,-1])
    plt.title('Frame %03d' %(5*k))
    Save_Name = Fol_Gif_y +'Gif_img%06d.png' %k # set the output name
    fig.savefig(Save_Name,dpi=60) # save the plot
    plt.close(fig) # close the figure to avoid overcrowding
    images_y.append(imageio.imread(Save_Name)) # append the name into the list of images
    
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
    ax.set_ylim(-14,0)
    ax.set_xlim(y[0,0],y[-1,0])
    plt.title('Frame %03d' %(5*k))
    Save_Name = Fol_Gif_x +'Gif_img%06d.png' %k # set the output name
    fig.savefig(Save_Name,dpi=60) # save the plot
    plt.close(fig) # close the figure to avoid overcrowding
    images_x.append(imageio.imread(Save_Name)) # append the name into the list of images
imageio.mimsave(GIFNAME_X, images_x, duration=0.2) # create the gif
imageio.mimsave(GIFNAME_Y, images_y, duration=0.2) # create the gif

import shutil  # nice and powerfull tool to delete a folder and its content
# shutil.rmtree(Fol_Gif_x)
# shutil.rmtree(Fol_Gif_y)






