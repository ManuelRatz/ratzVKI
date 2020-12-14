# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:07:36 2020

@author: Manuel Ratz
@description : Script to test the smoothing on 1 velocity profile for a fall.
    Loads a 3d tensor of a fall that has been smoothed in time and along the
    vertical axis and smoothes it along the horizontal one using a sine
    transformation
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import matplotlib.pyplot as plt
import numpy as np

def smooth_horizontal_sin(profiles, x_local, Width):
    """
    Function to smooth the horizontal profiles using a sin transformation.
    For that we are computing a temporal base Psi of sin(x*n*pi/Width), n = 1,2,3,...
    from which we calculate a projection matrix via Psi*Psi.T.
    Multiplying this new matrix with the horizontal profiles gives a smoothed 
    version that satisfies the boundary conditions automatically.
    
    This makes the 'smoothn_horizontal' function obsolete

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.
    x_local : 1d np.array
        1d array containing the x coordinates of the velocity field in pixels.
    Width : int
        Width of the channel in pixels.

    Returns
    -------
    smoothed_array : 3d np.array
        3d Tensor containing one smoothed velocity component of the field of 
        every timestep.

    """
    # define the n'th base function
    def basefunc(x, n, width, norm):                                                                                    
        return np.sin(n*x*np.pi/width)/norm  
    # initialize Psi, we remove 5 degrees of freedom with this
    Psi = np.zeros((x_local.shape[0],x_local.shape[0]-5))
    # calculate the norm
    norm = np.sqrt(x_local[-1]/2)
    # fill the Psi columns with the sin profiles
    for i in range(1, Psi.shape[1]+1):
        Psi[:,i-1] = basefunc(x_local, i, x_local[-1], norm)
    # calculate the projection
    projection = np.matmul(Psi,Psi.T)
    # create a dummy to fill
    smoothed_array = np.zeros(profiles.shape)
    # iterate in time
    for i in range(0, profiles.shape[0]):
        # iterate along y axis
        for j in range(0, profiles.shape[1]):
            # get the profile along x
            prof_hor = profiles[i,j,:]
            # check that we do not have one containing 1000s
            smoothed_array[i,j,:] = np.matmul(projection, prof_hor)*12
            # the multiplication by 8 is required for some reason to get
            # the correct magnitude, at the moment the reason for this is
            # unclear. For a Fall this has to be 12 for some reason.
    return smoothed_array, Psi    

# load the data
profile_unsmoothed = np.load('velocity_tensor.npy') # velocity profile
x_coordinates = np.load('x_coordinates.npy') # x coordinates (1d)

# smooth the profile
profile_smoothed, Psi = smooth_horizontal_sin(profile_unsmoothed, x_coordinates, x_coordinates[-1])

# checking the rank of Psi to make sure the columns are linearly independet
rank = np.linalg.matrix_rank(Psi)

# plot the result
fig, ax = plt.subplots()
ax.plot(x_coordinates, profile_unsmoothed[50,20,:], label = 'Unsmoothed')
ax.plot(x_coordinates, profile_smoothed[50,20,:], label = 'Smoothed')
ax.set_xlim(0, x_coordinates[-1])
ax.set_ylim(-80, 0)
ax.grid(b=True)
ax.legend(loc = 'upper center')
