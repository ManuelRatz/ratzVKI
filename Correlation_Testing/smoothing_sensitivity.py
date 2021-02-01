# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:19:27 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\new_processing')

import numpy as np
import os
import scipy.signal as sci 
import matplotlib.pyplot as plt
from smoothn import smoothn

def load_txt_files(Fol_In, Case):
    """
    Function to load the result files from the LeDaR interface detection. All
    files are already shifted to start at Frame0

    Parameters
    ----------
    Fol_In : str
        Location to the filepath were each case is stored.
    Case : str
        The given test case to plot.

    Returns
    -------
    pressure : 1d np.array
        Pressure in the facility in pa.
    h_avg : 1d np.array
        Average height of the interface in milimeter.
    ca_gauss : 1d np.array
        Contact angle from the gaussian fitting in degrees.
    ca_cosh : 1d np.array
        Contact angle from the hyperbolic cosine fit in degrees.
    fit_gauss : 1d np.array
        Average curvature of the gaussian fitting in 1/mm.
    fit_exp : 1d np.array
        Average curvature of the hyperbolic cosine fit in 1/mm.

    """
    # navigate to the data files
    Folder = os.path.join(Fol_In, 'data_files')
    # load each file, convert the angles to radians
    pressure = np.genfromtxt(os.path.join(Folder, Case + '_pressure.txt'))
    h_avg = np.genfromtxt(os.path.join(Folder, Case + '_h_avg.txt'))
    ca_gauss = np.genfromtxt(os.path.join(Folder, Case + '_ca_gauss.txt'))/180*np.pi
    ca_cosh = np.genfromtxt(os.path.join(Folder, Case + '_ca_cosh.txt'))/180*np.pi
    fit_gauss = np.genfromtxt(os.path.join(Folder, Case + '_gauss_curvature.txt'))
    fit_exp = np.genfromtxt(os.path.join(Folder, Case + '_cosh_curvature.txt'))
    h_cl = np.genfromtxt(os.path.join(Folder, Case + '_h_cl_r.txt'))
    # return all the files
    return pressure, h_avg, ca_gauss, ca_cosh, fit_gauss, fit_exp, h_cl


def save_data(file_name, dh, dhcl, ddh, ddhcl, theta):
    save_array = np.vstack([m.ravel() for m in [dh/1000, dhcl/1000, ddh/1000, ddhcl/1000, theta*180/np.pi]]).T
    np.savetxt(file_name, save_array, fmt = '%.5f', delimiter='\t')


def filter_signal(signal, cutoff_frequency, double_mirror = True):
    
    right_attach = 2*signal[-1]-signal[::-1]
    if double_mirror:
        left_attach = 2*signal[0]-signal[::-1]
        continued_signal = np.hstack((left_attach, signal, right_attach))
    else:
        continued_signal = np.hstack((signal, right_attach))

    windows = sci.firwin(numtaps = signal.shape[0]//10, cutoff = cutoff_frequency,\
                         window='hamming', fs = 500)
    # example_filtered = sci.filtfilt(b = windows, a = [1], x = continued_signal)
    example_filtered = sci.filtfilt(b = windows, a = [1], x = continued_signal, padlen = 7, padtype = 'even')
    clip_index = signal.shape[0]
    if double_mirror:
        clipped_signal = example_filtered[clip_index:2*clip_index]
    else:
        clipped_signal = example_filtered[:clip_index]
    return clipped_signal

def prepare_data(h_raw, h_cl_raw, ca_raw):
    vel_raw = np.gradient(h_raw, 0.002) 
    acc_raw = np.gradient(vel_raw, 0.002)
    vel_clean = filter_signal(vel_raw, 6)
    acc_clean = filter_signal(acc_raw, 6)
    
    vel_cl_raw = np.gradient(h_cl_raw, 0.002) 
    acc_cl_raw = np.gradient(vel_cl_raw, 0.002)
    vel_cl_clean = filter_signal(vel_cl_raw, 6)
    acc_cl_clean = filter_signal(acc_cl_raw, 6)
    
    return vel_clean, vel_cl_clean, acc_clean, acc_cl_clean

    

Fol_Save = 'Plots'
if not os.path.exists(Fol_Save):
    os.makedirs(Fol_Save)

""" Liquid properties """
# water:
mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724

# hfe
mu_h = 0.000667
rho_h = 1429.41965
sigma_h = 0.01391

t = np.linspace(0, 4, 2000, endpoint = False)
pressures_water = np.array([1000, 1250, 1500])
indices = np.array(['A', 'B', 'C'])
pressures_hfe = np.array([1500, 1750, 2000])

"""
# for pres in pressures_water:
#     Fol_Outer = 'C:\Anna\Rise\Water' + os.sep + 'P' + str(pres) + '_C30'
#     fig, ax = plt.subplots(figsize = (8,5))
#     for index in indices:
#         FOL = Fol_Outer + os.sep + index
#         Case = 'P' + str(pres) + '_C30_' + index
#         pressure, h_avg, ca_gauss, ca_cosh, curv_gauss, curv_cosh, h_cl = load_txt_files(FOL, Case)
#         ca_clean = filter_signal(ca_gauss, 10)
#         ax.plot(t, ca_clean*180/np.pi)
#         ax.grid(b = True)
#         ax.set_xlim(0, 4)
#         ax.set_xlabel('t')
#         ax.set_ylabel('Theta')
#         ax.set_title('Water comparison ' + str(pres) + ' Pa')
#         Name_Out = 'water_' + str(pres)
#         fig.savefig(Fol_Save + os.sep + Name_Out, dpi = 200)
        
# for pres in pressures_hfe:
#     Fol_Outer = 'C:\Anna\Rise\HFE' + os.sep + 'P' + str(pres)
#     fig, ax = plt.subplots(figsize = (8,5))
#     for index in indices:
#         FOL = Fol_Outer + os.sep + index
#         Case = 'P' + str(pres) + '_' + index
#         pressure, h_avg, ca_gauss, ca_cosh, curv_gauss, curv_cosh, h_cl = load_txt_files(FOL, Case)
#         ca_clean = filter_signal(ca_gauss, 10)
#         ax.plot(t, ca_clean*180/np.pi)
#         ax.grid(b = True)
#         ax.set_xlim(0, 4)
#         ax.set_xlabel('t')
#         ax.set_ylabel('Theta')
#         ax.set_title('HFE comparison ' + str(pres) + ' Pa')
#         Name_Out = 'hfe_' + str(pres)
#         fig.savefig(Fol_Save + os.sep + Name_Out, dpi = 200)
"""
for pres in pressures_water:
    Fol_Outer = 'C:\Anna\Rise\Water' + os.sep + 'P' + str(pres) + '_C30'
    for index in indices:
        FOL = Fol_Outer + os.sep + index
        Case = 'P' + str(pres) + '_C30_' + index
        pressure, h_avg, ca_gauss, ca_cosh, curv_gauss, curv_cosh, h_cl = load_txt_files(FOL, Case)
        
        v, v_cl, a, a_cl = prepare_data(h_avg, h_cl, ca_gauss)
        ca = filter_signal(ca_gauss, 8, double_mirror = False)
        Fol_Out = 'Data'
        if not os.path.exists(Fol_Out):
            os.makedirs(Fol_Out)
        File_Name = Fol_Out + os.sep + 'W_P'+str(pres) + '_' + index + '.txt'
        save_data(File_Name, v, v_cl, a, a_cl, ca)
        
for pres in pressures_hfe:
    Fol_Outer = 'C:\Anna\Rise\HFE' + os.sep + 'P' + str(pres)
    for index in indices:
        FOL = Fol_Outer + os.sep + index
        Case = 'P' + str(pres) + '_' + index
        pressure, h_avg, ca_gauss, ca_cosh, curv_gauss, curv_cosh, h_cl = load_txt_files(FOL, Case)
        
        v, v_cl, a, a_cl = prepare_data(h_avg, h_cl, ca_gauss)
        ca = filter_signal(ca_gauss, 8, double_mirror = False)
        Fol_Out = 'Data'
        if not os.path.exists(Fol_Out):
            os.makedirs(Fol_Out)
        File_Name = Fol_Out + os.sep + 'H_P'+str(pres) + '_' + index + '.txt'
        save_data(File_Name, v, v_cl, a, a_cl, ca)
  

#%%


sig = np.linspace(0, 1, 101)
sig_filter = filter_signal(sig, 8, double_mirror = True)
plt.plot(sig)
plt.plot(sig_filter)

pres = 1500
index = 'A'
Fol_Outer = 'C:\Anna\Rise\Water' + os.sep + 'P' + str(pres) + '_C30'
FOL = Fol_Outer + os.sep + index
Case = 'P' + str(pres) + '_C30' + '_' + index
pressure, h_avg, ca_gauss, ca_cosh, curv_gauss, curv_cosh, h_cl = load_txt_files(FOL, Case)


vel_raw = np.gradient(h_avg, 0.002) 
vel_clean = filter_signal(vel_raw, 6)
acc_raw = np.gradient(vel_clean, 0.002)
acc_clean = filter_signal(acc_raw, 6)



vel_cl_raw = np.gradient(h_cl, 0.002) 
vel_cl_clean = filter_signal(vel_cl_raw, 6, double_mirror = False)
acc_cl_raw = np.gradient(vel_cl_clean, 0.002)
acc_cl_clean = filter_signal(acc_cl_raw, 6, double_mirror = False)

# plt.figure()
# plt.plot(acc_cl_clean)
# plt.plot(acc_clean)

plt.figure()
plt.plot(np.gradient(h_cl[:100]))
plt.plot(np.gradient(h_avg[:100]))

plt.figure()
plt.plot(vel_clean[:100])
plt.plot(vel_cl_clean[:100])

plt.figure()
plt.plot(np.gradient(vel_clean, 0.002))
plt.plot(acc_clean)
# # v, v_cl, a, a_cl = prepare_data(h_avg, h_cl, ca_gauss)
ca = filter_signal(ca_gauss, 8, double_mirror = False)
# start = 0
# stop = start + 100
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 5))
# ax1.plot(vel_raw, lw = 0.3)
# ax1.plot(vel_clean, lw = 0.5)
# ax2.plot(acc_raw, lw = 0.3)
# ax2.plot(acc_clean, lw = 0.5)
# ax3.plot(ca_gauss, lw = 0.3)
# ax3.plot(ca, lw = 0.5)
# ax4.scatter(vel_clean, ca, s = 0.1)
# fig.savefig(Fol_Save + os.sep + Case + '.png', dpi = 400)

# plt.figure()
# plt.scatter(vel_cl_clean, ca, s = 0.1, marker = 'o', c = acc_clean, cmap = plt.cm.winter)