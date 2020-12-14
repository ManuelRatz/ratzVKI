# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 07:52:06 2020

@author: Manuel Ratz
"""
import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
import numpy as np
import matplotlib.pyplot as plt
import post_processing_functions as ppf
x = np.load('x.npy')
v = np.load('velocity.npy')

Width = x[-1]
# define the n'th base function
def basefunc(x, n, width, norm):
    return np.sin(n*x*np.pi/width)/norm

# initialize Psi
Psi = np.zeros((x.shape[0],20))
# calculate the norm
norm = np.sqrt(Width/2)
# fill the Psi columns with the sin profiles
for i in range(1, Psi.shape[1]+1):
    Psi[:,i-1] = basefunc(x, i, Width, norm)

# calculate the projection
projection = np.matmul(Psi,Psi.T)
# calculate the filtered velocity
u_filt = np.matmul(projection,v[100,20,:])*8
# plot to compare
fig, ax = plt.subplots()
ax.plot(x, v[100,20,:], label = 'Original')
ax.plot(x, u_filt, label = 'Cosine Transformed')
ax.legend()
ax.set_xlim(0, Width)