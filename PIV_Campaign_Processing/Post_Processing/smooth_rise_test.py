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
import os

# # load the data set and get the parameters of the images and the txt files
# ppf.set_plot_parameters(20, 15, 10)


# # for i in range(0, 10):
# order = 18
# x_tensor, y_tensor, u_tensor, v_tensor_raw, v_tensor_smoothed_smoothn = ppf.load_and_smooth(Fol_Sol, order = order)
# idx = 25
# fig, ax = plt.subplots()
# # ax.plot(x_tensor[idx,-1,:], v_tensor_smoothed_sin[idx,-1,:])
# ax.plot(x_tensor[idx,-1,:], v_tensor_raw[idx,-1,:])
# ax.plot(x_tensor[idx,-1,:], v_tensor_smoothed_smoothn[idx,-1,:])


# def flux_parameters(smoothed_profile, x, Scale):
#     """
#     Function to get the minimum and maximum flux for the mass flux

#     Parameters
#     ----------
#     smoothed_profile : 3d np.array
#         3d Tensor containing the smoothed vertical velocity component of the
#         field of every timestep. The invalid positions are filled with nans.
#     x : 2d np.array
#         x coordinates of the interrogation window centers in px.
#     Scale : float64
#         Conversion factor to go from px -> mm.

#     Returns
#     -------
#     q_max : int
#         Upper bound for the flux, set as maximum in the plots.
#     q_min : int
#         Lower bound for the flux, set as minimum in the plots
#     q_ticks : 1d np.array
#         The ticks for the y axis in the plots.

#     """
#     # get the flux for each valid column
#     q_tot = ppf.calc_flux(x[0,:], smoothed_profile, Scale)
#     maximum_q = np.nanmax(q_tot) 
#     minimum_q = np.nanmin(q_tot)
#     if (maximum_q-minimum_q) > 1500:
#         increments = 250        
#     elif (maximum_q-minimum_q) > 1000:
#         increments = 200
#     else:
#         increments = 100
#     divider_max = int(np.ceil(maximum_q/increments))
#     q_max = divider_max*increments
#     divider_min = int(np.floor(minimum_q/increments))
#     q_min = divider_min*increments
#     q_ticks = np.linspace(q_min, q_max, divider_max - divider_min+1)
#     return  q_max, q_min, q_ticks

# def quiver_parameters(smoothed_profile):
#     # increments of the vertical axis
#     increments = 50
#     maximum_v = np.nanmax(smoothed_profile)
#     minimum_v = np.nanmin(smoothed_profile)
#     divider_max = int(np.ceil(maximum_v/increments))
#     v_max = divider_max*increments
#     divider_min = int(np.floor(minimum_v/increments))
#     v_min = divider_min*increments
#     v_ticks = np.linspace(v_min, v_max, divider_max - divider_min+1)
#     return v_max, v_min, v_ticks

# qfield = ppf.calc_qfield(x_tensor, y_tensor, u_tensor, v_tensor_smoothed_smoothn)

# def qfield_parameters(qfield):
#     qfield_sorted = np.sort(qfield.ravel())
    
#     return Maximum_q, Minimum_q
# params = qfield_parameters(qfield)

# k = np.sort(qfield.ravel())
# valid = np.isfinite(k)
# k = k[valid]
# k = k[int(500):k.shape[0]-500]

# fig, ax = plt.subplots()
# ax.hist(k, bins = 150, range = (k[0],k[-1]))
# ax.set_ylim(0, 100)


def first_valid_row(array):
    nans = np.argwhere(np.isnan(array[:,5]))
    return nans.shape[0]

Fol_Sol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h2_f1200_1_p13_64_16'
Fol_Smo = ppf.get_smo_folder(Fol_Sol)
Height, Width = ppf.get_img_shape(ppf.get_raw_folder(Fol_Sol))

x_tensor = np.load(os.path.join(Fol_Smo, 'x_values.npy'))
y_tensor = np.load(os.path.join(Fol_Smo, 'y_values.npy'))
u_tensor = np.load(os.path.join(Fol_Smo, 'u_values.npy'))
v_tensor_raw = np.load(os.path.join(Fol_Smo, 'v_values_raw.npy'))
v_tensor_smo = np.load(os.path.join(Fol_Smo, 'v_values_smoothed.npy'))

import imageio
import os
IMAGES = []
Fol_Img = ppf.create_folder('tmp')
for i in range(0, 1):
    idx = 48 + i*4
    print(i)
    valid_row = first_valid_row(u_tensor[idx,:,:])
    x_pad = x_tensor[idx,valid_row:,:]
    y_pad = y_tensor[idx,valid_row:,:]
    u_pad = u_tensor[idx,valid_row:,:]
    v_pad = v_tensor_smo[idx,valid_row:,:]
    # v_raw_pad = v_tensor
    x = x_pad[:,1:-1]
    y = y_pad[:,1:-1]
    u = u_pad[:,1:-1]
    v = v_pad[:,1:-1]
    qfield = ppf.calc_qfield(x, y, u, v)
    x_pco, y_pco = ppf.shift_grid(x, y, padded = False)
    fig, ax = plt.subplots(figsize = (4.5, 8))
    cs = plt.pcolormesh(x_pco, y_pco, qfield, cmap = plt.cm.viridis, alpha = 1, vmin = -0.03, vmax = 0.02) # create the contourplot using pcolormesh
    ax.set_aspect('equal') # set the correct aspect ratio
    clb = fig.colorbar(cs, pad = 0.2, drawedges = False, alpha = 1) # get the colorbar
    clb.ax.set_title('Q Field \n [1/s$^2$]', pad=15) # set the colorbar title
    ax.set_xlim(0,Width)
    ax.set_ylim(0,Height)
    u_hp, v_hp = ppf.high_pass(u, v, 3, 3, padded=False)
    StpX = 1
    StpY = 1
    headlength = 5
    headwidth = 0.5*headlength
    width = 0.005
    ax.grid(b = False)
    ax.quiver(x[(x.shape[0]+1)%2::StpY,::StpX], y[(x.shape[0]+1)%2::StpY,::StpX],\
              u_hp[(x.shape[0]+1)%2::StpY,::StpX], v_hp[(x.shape[0]+1)%2::StpY,::StpX],\
              scale = 300, color = 'k', scale_units = 'height', units = 'width', width = width,\
              headlength = headlength, headwidth = headwidth, headaxislength = headlength)
    Name_Out = Fol_Img + os.sep + 'test%06d.png' %idx
    # fig.savefig(Name_Out, dpi = 80)
    # plt.close(fig)
    # IMAGES.append(imageio.imread(Name_Out))

# imageio.mimsave('test.gif', IMAGES, duration = 0.075)
# import shutil
# shutil.rmtree(Fol_Img)

#%%

