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

# load the data set and get the parameters of the images and the txt files
ppf.set_plot_parameters(20, 15, 10)
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f1200_1_p15_64_16'
Fol_Raw = ppf.get_raw_folder(Fol_Sol)
NX = ppf.get_column_amount(Fol_Sol) # number of columns
NY_max = ppf.get_max_row(Fol_Sol, NX) # maximum number of rows
Height, Width = ppf.get_img_shape(Fol_Raw) # height and width of the raw image in pixels

# give the first index and the number of files
Frame0 = 226
N_T = len(os.listdir(Fol_Sol + os.sep + 'data_files'))



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
        id1 = np.argmax(time_column != 100)
        # get the next index after id1 at which we have 100
        id2 = np.argmax(time_column[id1:] == 100)
        # check if the timeprofile is long enough
        if id2 < 50:
            # attempt to find a second non 100 index somewhere else
            id2_second_attempt = np.argmax(time_column[id1+id2+1:] == 100)
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
        every timestep.

    Returns
    -------
    work_array_smoothed : 3d np.array
        3d Tensor containing smoothed one velocity component of the field of 
        every timestep.

    """
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*100
    # iterate over each time step
    for j in range(0, smoothed_array.shape[0]):
        # find the first index at which we dont have 100
        valid_idx = np.argmax(profiles[j,:,1] != 100)
        # smooth along the vertical axis sliced after the valid idx
        smoothed_array[j,valid_idx:,:], Dum, Dum, Dum = smoothn(profiles[j,valid_idx:,:], s = 1, axis = 0, isrobust = True)
    return smoothed_array


def smooth_horizontal(profiles):
    """
    Function to smooth the profiles along the horizontal axis

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of 
        every timestep.

    Returns
    -------
    smoothed_array : T3d np.array
        3d Tensor containing smoothed one velocity component of the field of 
        every timestep.

    """
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*100
    # iterate in time
    for i in range(0, profiles.shape[0]):
        # iterate along y axis
        for j in range(0, profiles.shape[1]):
            # get the profile along x
            prof_hor = profiles[i,j,:]
            # check that we do not have one containing 100s
            if np.mean(prof_hor) > 50:
                continue
            else:
                # flip the signal
                dum_flip = np.flip(prof_hor)
                # pad the profile by extending it twice and flipping these extensions
                padded_profile = np.hstack((-dum_flip[:-1], prof_hor, -dum_flip[1:]))
                # smooth the new profile
                smoothed_pad, DUMMY, DUMMY, DUMMY = smoothn(padded_profile, s = 0.5)
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
print('Smoothing along the horizontal axis')
smo_tot = smooth_horizontal(smo_vert_freq)
print(time.time()-T4)


# animate a quick velocity profile comparison to see raw vs smoothed
ppf.set_plot_parameters(20, 15, 10)
t = 50 # start after 50 timesteps
y = -7 # seventh row from the bottom
import imageio
import os
fol = ppf.create_folder('tmp')
Gifname = 'Smoothing_comparison.gif'
listi = []
N_T = 100
for i in range(0, N_T):
    print('Image %d of %d' %((i+1),N_T))
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x[0,:], profiles_v[t+5*i,y,:], label = 'Unsmoothed')
    ax.plot(x[0,:], smo_tot[t+5*i,y,:], label = 'Smoothed')
    ax.legend(loc = 'lower right')
    ax.set_xlim(0, 266)
    ax.set_ylim(-10, 20)
    ax.grid(b = True)
    ax.set_xlabel('$x$[mm]')
    ax.set_ylabel('$v$[px/frame]')
    Name = fol + os.sep +'comp%06d.png' %i
    fig.savefig(Name, dpi = 65)
    listi.append(imageio.imread(Name))
    plt.close(fig)
imageio.mimsave(Gifname, listi, duration = 0.1)
import shutil
shutil.rmtree(fol)