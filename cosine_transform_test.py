# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 07:52:06 2020

@author: Manuel Ratz
"""

import numpy as np
import matplotlib.pyplot as plt

# array of x values
x_coord = np.load('x_coords.npy',)
# width of the channel (length of domain)
Width = x_coord[-1]

# define the n'th base function
def basefunc(x, n, width, norm):
    return np.sin(n*x*np.pi/width)/norm

# create a test signal consisting of the first 3 bases functions
u_test = basefunc(x_coord, 1, Width, 1) + 1*basefunc(x_coord, 2, Width, 1)\
    + 0.5*basefunc(x_coord, 3, Width, 1)

# initialize Psi
Psi = np.zeros((x_coord.shape[0],3))
# calculate the norm
norm = np.sqrt(Width/2)
# fill the Psi columns with the sin profiles
for i in range(1, Psi.shape[1]+1):
    Psi[:,i-1] = basefunc(x_coord, i, Width, norm)

# calculate the projection
projection = np.matmul(Psi,Psi.T)
# calculate the filtered velocity
u_filt = np.matmul(projection, u_test)
# plot to compare
fig, ax = plt.subplots()
ax.plot(x_coord, u_test, label = 'Original')
ax.plot(x_coord, u_filt, label = 'Cosine Transformed')
ax.legend()
ax.set_xlim(0, Width)