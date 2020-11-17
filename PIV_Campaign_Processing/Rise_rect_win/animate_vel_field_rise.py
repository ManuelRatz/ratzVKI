# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Code to animate velocity fields with quivers and contours
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

# set input and output folders
Fol_In_2 = 'G:\PIV_Processed\Images_Processed\Results_16_64_R_h1_f1200_1_p15' + os.sep
Fol_In_1 = 'G:\PIV_Processed\Images_Processed\Results_12_48_R_h1_f1200_1_p15' + os.sep
# Fol_Out = 'D:\PIV_Processed\Images_Postprocessed'+os.sep
# if not os.path.exists(Fol_Out):
#     os.mkdir(Fol_Out)


frame0 = 3
NAME = Fol_In_1 + os.sep + 'field_A%06d' % frame0 + '.txt'  # Check it out: print(Name)
data = np.genfromtxt(NAME)  # Here we have the four colums
nxny1 = data.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
X_S = data[:, 0]
Y_S = data[:, 1]
GRAD_Y = np.diff(Y_S)
IND_X = np.where(GRAD_Y != 0)
DAT = IND_X[0]
n_y1 = DAT[0] + 1
n_x1 = (nxny1 // (n_y1))  # Carefull with integer and float!
x1 = (X_S.reshape((n_x1, n_y1)))
y1 = (Y_S.reshape((n_x1, n_y1)))

NAME = Fol_In_2 + os.sep + 'field_A%06d' % frame0 + '.txt'  # Check it out: print(Name)
data = np.genfromtxt(NAME)  # Here we have the four colums
nxny2 = data.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
X_S = data[:, 0]
Y_S = data[:, 1]
GRAD_Y = np.diff(Y_S)
IND_X = np.where(GRAD_Y != 0)
DAT = IND_X[0]
n_y2 = DAT[0] + 1
n_x2 = (nxny2 // (n_y2))  # Carefull with integer and float!
x2 = (X_S.reshape((n_x2, n_y2)))
y2 = (Y_S.reshape((n_x2, n_y2)))

IDX1 = 20
IDX2 = 9
n_t = 10

for k in range(frame0, frame0+n_t):
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    NAME = Fol_In_1 + os.sep + 'field_A%06d' % (3*k+frame0) + '.txt' 
    DATA = np.genfromtxt(NAME)  # Here we have the four colums
    current_len = DATA.shape[0]
    V_X1 = DATA[:, 2]  # U component
    V_Y1 = DATA[:, 3]
    Mod = np.sqrt(V_X1 ** 2 + V_Y1 ** 2)
    u1 = (V_X1.reshape((n_x1, n_y1)))
    v1 = (V_Y1.reshape((n_x1, n_y1)))
    Magn = (Mod.reshape((n_x1, n_y1)))
    
    NAME = Fol_In_2 + os.sep + 'field_A%06d' % (3*k+frame0) + '.txt' 
    DATA = np.genfromtxt(NAME)  # Here we have the four colums
    current_len = DATA.shape[0]
    V_X2 = DATA[:, 2]  # U component
    V_Y2 = DATA[:, 3]
    Mod = np.sqrt(V_X2 ** 2 + V_Y2 ** 2)
    u2 = (V_X2.reshape((n_x2, n_y2)))
    v2 = (V_Y2.reshape((n_x2, n_y2)))
    Magn = (Mod.reshape((n_x2, n_y2)))
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x1[IDX1,:],v1[IDX1,:], label = 'Fine Grid')
    ax.scatter(x1[IDX1,:],v1[IDX1,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x2[IDX2,:],v2[IDX2,:], label = 'Rough Grid', c='r')
    ax.scatter(x2[IDX2,:],v2[IDX2,:], marker='x', s=(300./fig.dpi)**2, c='r')  
    ax.grid(b = True, lw = 2)
    ax.legend(loc = 'lower center', ncol = 3)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(-5,17.5)
    ax.set_xlim(0,270)
    plt.title('Frame %03d' %(3*k))






