# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:40:18 2019

@author: mendez
"""

import os  # This is to understand which separator in the paths (/ or \)

import matplotlib.pyplot as plt  # This is to plot things
import numpy as np  # This is for doing math

################## Post Processing of the PIV Field.

## Step 1: Read all the files and (optional) make a video out of it.

FOLDER = 'Results_PIV' + os.sep + 'Open_PIV_results_Inversion_24'
Fol_Out = 'Images_postprocessed' + os.sep + 'rise_inversion_24' + os.sep
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
Fol_Gif = 'Images_postprocessed' + os.sep + 'Gif_images' + os.sep
if not os.path.exists(Fol_Gif):
    os.mkdir(Fol_Gif)

n_t = 5  # number of steps.

# Read file number 10 (Check the string construction)
Name = FOLDER + os.sep + 'field_A%03d' % 1 + '.txt'  # Check it out: print(Name)
# Read data from a file
DATA = np.genfromtxt(Name)  # Here we have the four colums
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

# fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
# # Plot Contours and quiver
# plt.contourf(Xg * 1000, Yg * 1000, Magn)
# plt.quiver(X_S * 1000, Y_S * 1000, V_X, V_Y)

###### Step 2: Compute the Mean Flow and the standard deviation.
# The mean flow can be computed by assembling first the DATA matrices D_U and D_V
D_U = np.zeros((n_s, n_t))
D_V = np.zeros((n_s, n_t))
# Loop over all the files: we make a giff and create the Data Matrices
D_U = np.zeros((n_x * n_y, n_t))  # Initialize the Data matrix for U Field.
D_V = np.zeros((n_x * n_y, n_t))  # Initialize the Data matrix for V Field.
# Profile_V = np.zeros((len(Xg[0,:]), n_t))
for k in range(0, n_t):
    # Read file number 10 (Check the string construction)
    Name = FOLDER + os.sep + 'field_A%03d' % (k) + '.txt'  # Check it out: print(Name)
    # We prepare the new name for the image to export
    NameOUT = Fol_Out + os.sep + 'Im%03d' % (k) + '.png'  # Check it out: print(Name)
    # Read data from a file
    DATA = np.genfromtxt(Name)  # Here we have the four colums
    V_X = DATA[:, 2]  # U component
    V_Y = DATA[:, 3]  # V component
    # Put both components as fields in the grid
    Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
    Vxg = (V_X.reshape((n_x, n_y)))
    Vyg = (V_Y.reshape((n_x, n_y)))
    Magn = (Mod.reshape((n_x, n_y)))
    # Prepare the D_MATRIX
    D_U[:, k] = V_X
    D_V[:, k] = V_Y
    # Open the figure
    fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
    # Or you can plot it as streamlines
    plt.contourf(Xg *1000 , Yg*1000 , Magn)
    # One possibility is to use quiver
    STEPx = 1
    STEPy = 1
    
    plt.quiver(Xg[::STEPx, ::STEPy] * 1000, Yg[::STEPx, ::STEPy] * 1000,
               Vxg[::STEPx, ::STEPy], Vyg[::STEPx, ::STEPy], color='k', scale = 1000)  # Create a quiver (arrows) plot
    plt.rc('text', usetex=True)  # This is Miguel's customization
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # ax is an object, which we could get also using ax=plt.gca() 
    # We can now modify all the properties of this obect 
    # In this exercise we follow an object-oriented approach. These
    # are all the properties we modify
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$', fontsize=18)
    ax.set_ylabel('$y[mm]$', fontsize=18)
    #   ax.set_title('Velocity Field via TR-PIV',fontsize=18)
    # ax.set_xticks(np.arange(0, 201, 100))
    # ax.set_yticks(np.arange(5, 30, 5))
    #   ax.set_xlim([0,43])
    #   ax.set_ylim(0,28)
    #   ax.invert_yaxis() # Invert Axis for plotting purpose
    # Observe that the order at which you run these commands is important!
    # Important: we fix the same c axis for every image (avoid flickering)
    plt.axis('off')
    plt.clim(0, 10)
    plt.colorbar()  # We show the colorbar
    plt.savefig(NameOUT, dpi=800)
    plt.close(fig)
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(Xg[0,:],Vyg[30,:])
    plt.scatter(Xg[0,:],Vyg[30,:])
    ax.set_xlim(0,260)
    ax.set_ylim(-10,10)
    Save_Name = Fol_Gif +'Gif_img%03d.png' %k
    fig.savefig(Save_Name,dpi=100)
    plt.close(fig)

# import imageio
# GIFNAME = 'Giff_Velocity.gif'
# images = []

# for k in range(0, n_t):
#     FIG_NAME = Fol_Gif +'Gif_img%03d' %k + '.png'
#     images.append(imageio.imread(FIG_NAME))
    
# # Now we can assembly the video and clean the folder of png's (optional)
# imageio.mimsave(GIFNAME, images, duration=0.05)
# import shutil  # nice and powerfull tool to delete a folder and its content
# shutil.rmtree(Fol_Gif)