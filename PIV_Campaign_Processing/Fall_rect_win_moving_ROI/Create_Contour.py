# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:00:50 2020

@author: manue
"""
import os  # This is to understand which separator in the paths (/ or \)
import imageio # for rendering the gif
import matplotlib.pyplot as plt  # This is to plot things
import numpy as np  # This is for doing math

run = 'F_h1_f1200_1_q_24_24'

# set input and output folders
Fol_In = 'D:\PIV_Processed\Images_Processed\Open_PIV_results_' + run + os.sep

NAME = Fol_In + os.sep + 'field_A%06d' % 663+ '.txt'  # Check it out: print(Name)
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
V_X = data[:, 2]  # U component
V_Y = data[:, 3]
Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
u = (V_X.reshape((n_x, n_y)))
v = (V_Y.reshape((n_x, n_y)))
Magn = (Mod.reshape((n_x, n_y)))


fig, ax = plt.subplots(figsize=(5,10))
STEPx = 1
STEPy = 8
cs = ax.contourf(x,y,v, cmap = plt.cm.viridis)
plt.quiver(x[::STEPy, ::STEPx], y[::STEPy, ::STEPx], u[::STEPy, ::STEPx], v[::STEPy, ::STEPx],\
           color='k', scale =40, width=0.004,headwidth=10, headaxislength = 6)
ax.set_aspect('equal')  # Set equal aspect ratio
ax.set_xticks(np.arange(10, 272, 48.8))
ax.set_xticklabels(np.arange(0, 260, 50))
plt.colorbar(cs)
cs.set_clim(-12,0)
plt.show()
fig.savefig('Contourplot.png',dpi = 400)













