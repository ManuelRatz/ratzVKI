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
from smoothn import smoothn
import post_processing_functions as ppf
import os

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
    Psi = np.zeros((x_local.shape[0],15))
    # calculate the norm
    norm = np.sqrt(x_local[-1]/2)
    # fill the Psi columns with the sin profiles
    for i in range(1, Psi.shape[1]+1):
        Psi[:,i-1] = -basefunc(x_local, i, x_local[-1], norm)
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
Fol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS'
runs = os.listdir(Fol)
# for i in range(4,5):
for i in range(0, len(runs)):
    run = runs[i]
    Fol_Sol = Fol + os.sep + run
    print(run)

    x_tensor, y_tensor, u_tensor, v_tensor, smoothed_sine, smoothed_smoothn = ppf.load_and_smooth(Fol_Sol)
    x_coordinates = x_tensor[0,-1,:]
    
    # smooth the profile
    # profile_smoothed, Psi = smooth_horizontal_sin(profile_unsmoothed, x_coordinates, x_coordinates[-1])
    
    # # checking the rank of Psi to make sure the columns are linearly independet
    # rank = np.linalg.matrix_rank(Psi)
    
    # # plot the result
    # fig, ax = plt.subplots()
    # ax.plot(x_coordinates, profile_unsmoothed[50,20,:], label = 'Unsmoothed')
    # ax.plot(x_coordinates, profile_smoothed[50,20,:], label = 'Smoothed')
    # ax.set_xlim(0, x_coordinates[-1])
    # ax.set_ylim(-80, 0)
    # ax.grid(b=True)
    # ax.legend(loc = 'upper center')
    
    #%% Miguel Check.
    # We will look for the coefficients in the coarse mesh,
    # but then use the in the fine mesh (fundamentals of 'super-resolution')
    # We create the basis; I put 15 vectors
    N_V=30
    # Initialize the matrix
    Psi_S_Coarse=np.zeros([x_coordinates.shape[0],N_V])
    L=x_coordinates.max()
    # Loop over columns for construction
    for r in range(1,N_V):
        psi=np.sin(np.pi*x_coordinates*r/(L))
        Psi_S_Coarse[:,r-1]=psi/np.linalg.norm(psi)
    
    # # Show the bases
    # plt.plot(Psi_S_Coarse[:,0])
    # plt.plot(Psi_S_Coarse[:,1])
    # plt.plot(Psi_S_Coarse[:,2])
    
    # We orthonormalize it:
    q, r = np.linalg.qr(Psi_S_Coarse)
    # plt.plot(q[:,0])
    # plt.plot(q[:,1])
    # plt.plot(q[:,2])
    
    
    #%% We compute the coefficients of the projection,
    # based on an approximation
    q_tilde=q[:,0:18]
    
    U=smoothed_smoothn[150,20,:]
    
    dum_flip = np.flip(U)
    # pad the profile by extending it twice and flipping these extensions
    padded_profile = np.hstack((-dum_flip[:-1], U, -dum_flip[1:]))
    smo_middle, Dummy, Dummy, Dummy = smoothn(padded_profile, s = 0.5, isrobust = False)
    pad_smo = smo_middle[x_coordinates.shape[0]-1:2*x_coordinates.shape[0]-1]
    
    # plt.plot(pad_smo)
    # plt.plot(U)
    check=np.matmul(q_tilde.T, q_tilde)
    Coeffs=np.matmul(q_tilde.T, U)
    U_tilde=np.matmul(q_tilde,Coeffs)
    fig, ax = plt.subplots()
    # ax.plot(x_coordinates,U)
    ax.plot(x_coordinates,U_tilde)
    # ax.set_ylim(-350, 0)
    ax.set_xlim(0, x_coordinates[-1])
    print('##########################################')
    
    # x = np.copy(x_coordinates)
    # Psi_S_Fine = np.zeros((x.shape[0],23))
    # for r in range(1, 23):
    #     psi = np.sin(np.pi*x*r/L)
    #     Psi_S_Fine[:,r-1] = psi/np.linalg.norm(psi)
    # Psi_S_Fine = Psi_S_Fine[:,:23]
    # q_new, r_new = np.linalg.qr(Psi_S_Fine)
    
    # check2 = np.matmul(q_new.T,q_new)
    # U_fine = np.matmul(q_new,Coeffs)
    
    # fig, ax = plt.subplots()
    # ax.plot(x,U_fine)