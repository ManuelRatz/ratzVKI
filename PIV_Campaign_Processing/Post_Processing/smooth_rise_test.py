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


# load the data set and get the parameters of the images and the txt files
ppf.set_plot_parameters(20, 15, 10)
Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h2_f1200_1_p13_64_16'

# for i in range(0, 10):
order = 18
x_tensor, y_tensor, u_tensor, v_tensor_raw, v_tensor_smoothed_sin, v_tensor_smoothed_smoothn = ppf.load_and_smooth(Fol_Sol, order = order)
idx = 8
fig, ax = plt.subplots()
ax.plot(x_tensor[idx,-1,:], v_tensor_smoothed_sin[idx,-1,:])
ax.plot(x_tensor[idx,-1,:], v_tensor_raw[idx,-1,:])
ax.plot(x_tensor[idx,-1,:], v_tensor_smoothed_smoothn[idx,-1,:])


#%%
t = 100
x = x_tensor[t,:,:]
y = y_tensor[t,:,:]
u = u_tensor[t,:,:]
v = v_tensor_smoothed_smoothn[t,:,:]
qfield = ppf.calc_qfield(x, y, u, v)
x_pco, y_pco = ppf.shift_grid(x, y)

fig, ax = plt.subplots(figsize = (4.5, 8))
cs = plt.pcolormesh(x,y,qfield, cmap = plt.cm.Blues, alpha = 0.7, visible = True, linewidth = 0.000005) # create the contourplot using pcolormesh
ax.set_aspect('equal') # set the correct aspect ratio
clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
# clb.set_ticks(np.linspace(-100, 0, 6)) # set the colorbarticks
clb.ax.set_title('Q Field \n [1/s$^2$]', pad=15) # set the colorbar title

u, v = ppf.high_pass(u, v, 3, 3)
StpX = 1
StpY = 1
ax.grid(b = False)
ax.quiver(x[::StpY,::StpX], y[::StpY,::StpX], u[::StpY,::StpX], v[::StpY,::StpX], scale = 70, color = 'red')
fig.savefig('test.png', dpi = 80)

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