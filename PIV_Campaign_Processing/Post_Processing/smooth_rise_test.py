# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:59:46 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import post_processing_functions as ppf
import matplotlib.pyplot as plt
import scipy.signal as sci 
import os
from smoothn import smoothn

# this is a fill dummy, meaning all invalid positions are filled with 1000
Fill_Dum = 1000
# load the data set and get the parameters of the images and the txt files
ppf.set_plot_parameters(20, 15, 10)
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h2_f1200_1_p13_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
NX = ppf.get_column_amount(Fol_Sol) # number of columns
NY_max = ppf.get_max_row(Fol_Sol, NX) # maximum number of rows
Height, Width = ppf.get_img_shape(Fol_Raw) # image width in pixels
Scale = Width/5 # scaling factor in px/mm
Dt = 1/ppf.get_frequency(Fol_Raw) # time between images
Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s

# give the first index and the number of files
Frame0 = ppf.get_frame0(Fol_Sol)
N_T = ppf.get_NT(Fol_Sol)



import time
Start = time.time()
print('Loading Velocity Fields')
"""
Load the velocity profiles and put them into the 3d tensor in which all profiles are stored.

For that we load the velocity field for each time step and add 0 padding at the edges.
Afterwards we fill the velocity field to the top with 100s because not every index
has the full region of interest, so in order to be able to copy it into the 3d tensor
we have to add dummys at the top. In case something messes up, we use 100 and not 
np.nan because firwin cannot deal with nans
"""
# initialize the velocity profile tensors, the width is +2 because we have 0 padding
profiles_u = np.zeros((N_T, NY_max, NX+2))
profiles_v = np.zeros((N_T, NY_max, NX+2))



for i in range(0, N_T):
    Load_Index = Frame0 + i # load index of the current file
    x, y, u, v, ratio, mask = ppf.load_txt(Fol_Sol, Load_Index, NX) # load the data from the txt
    x, y, u, v, ratio, valid, invalid = ppf.filter_invalid(x, y, u, v, ratio, mask, valid_thresh = 0.5) # filter invalid rows
    x, y, u, v = ppf.pad(x, y, u, v, Width) # 0 pad at the edges
    u = u*Factor
    v = v*Factor
    u, v = ppf.fill_dummy(u, v, NY_max) # fill the 2d velocity array with 100s
    profiles_v[i,:,:] = v # store in the tensor
    profiles_u[i,:,:] = u # store in the tensor
T2 = time.time()
print(T2-Start)
#%%
def smooth_frequency(profiles):
    """
    Function to smooth the velocity profiles along the time axis in the frequency
    domain. For this we are using filtfilt and firwin from scipy. The principle
    as follows:
        1) For each row and column in the velocity field we extract the row in
           time (henceforth called timerow).
        2) Because not all timerows have data in them (some have 100s) we find
           the first index not equal to 100, this is our starting index. We then
           search for the second index not equal to 100. If there is none, all
           other points are valid, so we take all of those points after the first
        3) If there is a second one we check the number of steps in between.
           If it is below 50, we search again after the second peak as the array
           is either too short (then nothing happens) or we get a longer section
           that is valid for our smoothing
        4) We then smooth the extracted profile in time using a Firwin length
           of 1/30th of the profile length. The smoothed array is then returned

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of every timestep.
        The axiies are as follows:
            0 - time
            1 - y (vertical)
            2 - x (horizontal)

    Returns
    -------
    smoothed_array : 3d np.array
        3d Tensor containing one smoothed velocity component of the field of 
        every timestep.

    """
    def smooth_profile(tmp_prof):
        """
        Function to smooth a 1d profile in time using firwin and filtfilt.
        IMPORTANT: The profile must not contain nans otherwise filtfilt will
        only amplify this problem

        Parameters
        ----------
        tmp_prof : 1d np.array
            Array containing one velocity component at one position in time.

        Returns
        -------
        prof_filtered : 1d np.array
            Array containing one smoothed velocity component at one position in time.

        """
        # select the windows
        filter_poly = sci.firwin(tmp_prof.shape[0]//30, 0.05 , window='hamming', fs = 100)
        # filter with filtfilt
        prof_filtered = sci.filtfilt(b = filter_poly, a = [1], x = tmp_prof, padlen = 10, padtype = 'even')
        # return the filtered profile
        return prof_filtered
    def extract_profile(time_column):
        """
        Function to extract the longest velocity profile in time from the
        current time step.

        Parameters
        ----------
        time_column : 1 np.array of length N_T (total number of images)
            DESCRIPTION.

        Returns
        -------
        tmp_prof : 1d np.array
            Temporary array containing a velocity profile in time at a fixed
            point in space.
        id_min : int
            Lower index at which the array is sliced.
        id_max : int
            Upper index at which the array is sliced.

        """
        # get the first index at which we dont have 100
        id1 = np.argmax(time_column < 0.8*Fill_Dum)
        # get the next index after id1 at which we have 100
        id2 = np.argmax(time_column[id1:] > 0.8*Fill_Dum)
        # check if the timeprofile is long enough
        if id2 < 50:
            # attempt to find a second non 100 index somewhere else
            id2_second_attempt = np.argmax(time_column[id1+id2+1:] > 0.8*Fill_Dum)
            # np.argmax returns 0 if nothing was found, so test this here
            if id2_second_attempt == 0:
                # if true we set the second index to the last of the time_column
                id2_second_attempt = time_column.shape[0]-1
            # set minimum and maximum index accordingly
            id_min = id1 + id2
            id_max = id2_second_attempt
        else:
            # set minimum and maximum index accordingly
            id_min = id1
            id_max = id1 + id2
        # slice the time profile
        tmp_prof = time_column[id_min:id_max]
        # return the sliced array and the two indices
        return tmp_prof, id_min, id_max
    
    # start by copying the array to avoid editing the original array
    smoothed_array = np.copy(profiles)
    # iterate across the y axis
    for j in range(0, smoothed_array.shape[1]):
        # iterate across the x axis
        for i in range(0, smoothed_array.shape[2]):
            # extract the longest profile in time
            tmp_prof, id1, id2 = extract_profile(smoothed_array[:,j,i])
            # check if it is long enough for filtfilt
            if tmp_prof.shape[0] < 100:
                continue
            # smooth the respective profile in time and copy into the dummy array
            smoothed_array[id1:id2,j,i] = smooth_profile(tmp_prof)
    # return the smoothed array
    return smoothed_array

def smooth_vertical(profiles):
    """
    Function to smooth the 3d tensor along the vertical axis

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.

    Returns
    -------
    work_array_smoothed : 3d np.array
        3d Tensor containing smoothed one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s. 

    """
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*Fill_Dum
    # iterate over each time step
    for j in range(0, smoothed_array.shape[0]):
        # find the first index at which we dont have 100
        valid_idx = np.argmax(np.abs(profiles[j,:,1]) < 0.8*Fill_Dum)
        # smooth along the vertical axis sliced after the valid idx
        smoothed_array[j,valid_idx:,:], Dum, Dum, Dum = smoothn(profiles[j,valid_idx:,:], s = 1, axis = 0, isrobust = True)
    return smoothed_array


def smoothn_horizontal(profiles):
    """
    Function to smooth the profiles along the horizontal axis

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.

    Returns
    -------
    smoothed_array : 3d np.array
        3d Tensor containing one smoothed velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.

    """
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*Fill_Dum
    # iterate in time
    for i in range(0, profiles.shape[0]):
        # iterate along y axis
        for j in range(0, profiles.shape[1]):
            # get the profile along x
            prof_hor = profiles[i,j,:]
            # check that we do not have one containing 100s
            if np.mean(prof_hor) > 0.8*Fill_Dum:
                continue
            else:
                # flip the signal
                dum_flip = np.flip(prof_hor)
                # pad the profile by extending it twice and flipping these extensions
                padded_profile = np.hstack((-dum_flip[:-1], prof_hor, -dum_flip[1:]))
                # smooth the new profile
                smoothed_pad, DUMMY, DUMMY, DUMMY = smoothn(padded_profile, s = 0.1)
            # copy the sliced, smoothed profile into the tensor
            smoothed_array[i,j,:] = smoothed_pad[v.shape[1]-1:2*v.shape[1]-1]
    return smoothed_array      

"""
We now smooth along all three axiies, first timewise (axis 0) in the frequency
domain, then along the vertical direction (axis 1) with robust smoothn and finally
along the horizontal axis, again with smoothn.
"""
print('Smoothing along the time axis')
smo_freq = smooth_frequency(profiles_v)
T3 = time.time()
print(T3-T2)
print('Smoothing along the vertical axis')
smo_vert_freq = smooth_vertical(smo_freq)
T4 = time.time()
print(T4-T3)
# print('Smoothing along the horizontal axis with smoothn')
# #%%
# smo_tot = smoothn_horizontal(smo_vert_freq)

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
    Psi = np.zeros((x_local.shape[0],x_local.shape[0]-10))
    # calculate the norm
    norm = np.sqrt(x_local[-1]/2)
    # fill the Psi columns with the sin profiles
    for i in range(1, Psi.shape[1]+1):
        Psi[:,i-1] = basefunc(x_local, i, x_local[-1], norm)
    # calculate the projection
    projection = np.matmul(Psi,Psi.T)
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*Fill_Dum
    # iterate in time
    for i in range(0, profiles.shape[0]):
        # iterate along y axis
        for j in range(0, profiles.shape[1]):
            # get the profile along x
            prof_hor = profiles[i,j,:]
            # check that we do not have one containing 100s
            if np.mean(prof_hor) > 0.8*Fill_Dum:
                continue
            else:
                smoothed_array[i,j,:] = np.matmul(projection, prof_hor)*8
                # the multiplication by 8 is required for some reason to get
                # the correct magnitude, at the moment the reason for this is
                # unclear. For a Fall this has to be 12 for some reason.
    return smoothed_array, Psi    
T5 = time.time()
print(T5-T4)
print('Smoothing along the horizontal axis with cosine transformation')
smo_tot_cos, Psi = smooth_horizontal_sin(smo_vert_freq, x[0,:], Width)
print(time.time()-T5)

inv = smo_tot_cos > 0.8*Fill_Dum
smo_tot_cos[inv] = np.nan


def flux_parameters(smoothed_profile, x, Scale):
    """
    Function to get the minimum and maximum flux for the mass flux

    Parameters
    ----------
    smoothed_profile : 3d np.array
        3d Tensor containing the smoothed vertical velocity component of the
        field of every timestep. The invalid positions are filled with nans.
    x : 2d np.array
        x coordinates of the interrogation window centers in px.
    Scale : float64
        Conversion factor to go from px -> mm.

    Returns
    -------
    q_max : int
        Upper bound for the flux, set as maximum in the plots.
    q_min : int
        Lower bound for the flux, set as minimum in the plots
    q_ticks : 1d np.array
        The ticks for the y axis in the plots.

    """
    # get the flux for each valid column
    q_tot = ppf.calc_flux(x[0,:], smoothed_profile, Scale)
    maximum_q = np.nanmax(q_tot) 
    minimum_q = np.nanmin(q_tot)
    if (maximum_q-minimum_q) > 1500:
        increments = 250        
    elif (maximum_q-minimum_q) > 1000:
        increments = 200
    else:
        increments = 100
    divider_max = int(np.ceil(maximum_q/increments))
    q_max = divider_max*increments
    divider_min = int(np.floor(minimum_q/increments))
    q_min = divider_min*increments
    q_ticks = np.linspace(q_min, q_max, divider_max - divider_min+1)
    return  q_max, q_min, q_ticks
qmin, qmax, q_ticks = flux_parameters(smo_tot_cos, x, Scale)

def profile_parameters(smoothed_profile):
    # increments of the vertical axis
    increments = 50
    maximum_v = np.nanmax(smoothed_profile)
    minimum_v = np.nanmin(smoothed_profile)
    divider_max = int(np.ceil(maximum_v/increments))
    v_max = divider_max*increments
    divider_min = int(np.floor(minimum_v/increments))
    v_min = divider_min*increments
    v_ticks = np.linspace(v_min, v_max, divider_max - divider_min+1)
    if v_min - minimum_v > -50:
        v_min = v_min - 50
    return v_max, v_min, v_ticks

vmin, vmax, v_ticks = profile_parameters(smo_tot_cos)
t0 = 2707
for i in range(0, 1):
    fig, ax = plt.subplots()
    ax.plot(x[0,:],profiles_v[t0+5*i,30,:], label = 'raw')
    ax.plot(x[0,:],smo_vert_freq[t0+5*i,30,:], label = 'freq and vert')
    ax.plot(x[0,:],smo_freq[t0+5*i,30,:], label = 'freq')
    ax.plot(x[0,:],smo_tot_cos[t0+5*i,30,:], label = 'total')
    ax.legend()
# ax.plot(smo_tot_cos[5,60,:])
# ax.plot(smo_tot_cos[5,100,:])
# animate a quick velocity profile comparison to see raw vs smoothed
# ppf.set_plot_parameters(20, 15, 10)
# t = 100 # start after 50 timesteps
# y = -2 # second row from the bottom
# import imageio
# import os
# fol = ppf.create_folder('tmp')
# Gifname = 'Smoothing_comparison.gif'
# listi = []
# N_T = 100
# for i in range(0, N_T):
#     print('Image %d of %d' %((i+1),N_T))
#     fig, ax = plt.subplots(figsize = (8,5))
#     ax.plot(x[0,:], profiles_v[t+6*i,y,:], label = 'Unsmoothed', c = 'k')
#     ax.plot(x[0,:], smo_tot[t+6*i,y,:], label = 'Smoothn', c = 'lime')
#     ax.plot(x[0,:], smo_tot_cos[t+6*i,y,:], label = 'Sin Transformation', c = 'r')
#     ax.legend(loc = 'lower right')
#     ax.set_xlim(0, Width)
#     ax.set_ylim(-200, 400)
#     ax.grid(b = True)
#     ax.set_xlabel('$x$[mm]')
#     ax.set_ylabel('$v$[px/frame]')
#     Name = fol + os.sep +'comp%06d.png' %i
#     fig.savefig(Name, dpi = 65)
#     listi.append(imageio.imread(Name))
#     plt.close(fig)
# imageio.mimsave(Gifname, listi, duration = 0.05)
# import shutil
# shutil.rmtree(fol)