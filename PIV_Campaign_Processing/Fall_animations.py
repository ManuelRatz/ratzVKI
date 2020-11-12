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
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=20)   # fontsize of the figure title
plt.rc('text', usetex=False)     # use latex for the text
plt.rc('font', family='serif')   # serif as text font
plt.rc('axes', grid=True)        # enable the grid
plt.rc('savefig', dpi = 100)     # set the dpi for saving figures

# set input and output folders
Fol_In = 'C:\\Users\manue\Desktop\\tmp_processed\Open_PIV_results_F_h2_f1000_1_q_24_24'+os.sep
Fol_Out = 'C:\\Users\manue\Desktop\\tmp_processed\post'+os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
Fol_Gif = 'C:\\Users\manue\Desktop\\tmp_processed\post'+os.sep
if not os.path.exists(Fol_Gif):
    os.mkdir(Fol_Gif)

GIFNAME = Fol_Gif + '..' + os.sep + 'constant_y.gif' # name of the gif 
images = [] # empty list for the image names

# Read file number 10 (Check the string construction)
NAME = Fol_In + os.sep + 'field_A%06d' % (0) + '.txt'  # Check it out: print(Name)
# Read data from a file
DATA = np.genfromtxt(NAME)  # Here we have the four colums
nxny = DATA.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
n_s = 2 * nxny
## 1. Reconstruct Mesh from file
X_S = DATA[:, 0]
Y_S = DATA[:, 1]
# Number of n_X/n_Y from forward differences
GRAD_Y = np.diff(Y_S)
# Depending on the reshaping performed, one of the two will start with
# non-zero gradient. The other will have zero gradient only on the change.
IND_X = np.where(GRAD_Y != 0)
DAT = IND_X[0]
n_x = DAT[0] + 1
# Reshaping the grid from the data
n_y = (nxny // (n_x))  # Carefull with integer and float!

n_t = int(len(os.listdir(Fol_In))/2)  # number of steps.

# initialize the data matrix for one row in space for the time filtering
d_1 = np.zeros((n_t, n_x))

# n_t = 5
for k in range(0, n_t):
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    # Read file number 10 (Check the string construction)
    NAME = Fol_In + os.sep + 'field_A%06d' % (k) + '.txt'  # Check it out: print(Name)
    # We prepare the new name for the image to export
    NameOUT = Fol_Out + os.sep + 'Im%06d' % (k) + '.png'  # Check it out: print(Name)
    # Read data from a file
    DATA = np.genfromtxt(NAME)  # Here we have the four colums
    nxny = DATA.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    n_s = 2 * nxny
    ## 1. Reconstruct Mesh from file
    X_S = DATA[:, 0]
    Y_S = DATA[:, 1]
    # Number of n_X/n_Y from forward differences
    GRAD_Y = np.diff(Y_S)
    # Depending on the reshaping performed, one of the two will start with
    # non-zero gradient. The other will have zero gradient only on the change.
    IND_X = np.where(GRAD_Y != 0)
    DAT = IND_X[0]
    n_y = DAT[0] + 1
    # Reshaping the grid from the data
    n_x = (nxny // (n_y))  # Carefull with integer and float!
    Xg = (X_S.reshape((n_x, n_y)))
    Yg = (Y_S.reshape((n_x, n_y)))  # This is now the mesh! 60x114.
    # Reshape also the velocity components
    V_X = DATA[:, 2]  # U component
    V_Y = DATA[:, 3]  # V component
    # Put both components as fields in the grid
    Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
    Vxg = (V_X.reshape((n_x, n_y)))
    Vyg = (V_Y.reshape((n_x, n_y)))
    Magn = (Mod.reshape((n_x, n_y)))
    
    d_1[k,:] = Vyg[-2]
    roi = (30,40)
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(Xg[0,:],Vyg[-2,:]) # plot the second lowest velocity profile
    plt.scatter(Xg[0,:],Vyg[-2,:]) # same with points
    ax.set_xlim(Xg[0,0],Xg[0,-1]) # set xlimit
    ax.set_ylim(-24,0) # set ylimit
    ax.invert_yaxis() # invert Axis for plotting purpose
    # plt.quiver(Xg[roi[0]:roi[1]], Yg[roi[0]:roi[1]], Vxg[roi[0]:roi[1]], Vyg[roi[0]:roi[1]]\
    #             ,width=0.001,headwidth=3,scale = 500)
    plt.title('Image %d of %d' %((k+1), n_t)) # set the title to keep count
    Save_Name = Fol_Gif +'Gif_img%03d.png' %k # set the output name
    # plt.show()
    fig.savefig(Save_Name,dpi=100) # save the plot
    plt.close(fig) # close the figure to avoid overcrowding
    images.append(imageio.imread(Save_Name)) # append the name into the list of images

imageio.mimsave(GIFNAME, images, duration=0.1) # create the gif

# import shutil  # nice and powerfull tool to delete a folder and its content
# shutil.rmtree(Fol_Gif)