# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:34:51 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\new_processing')
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as sci 
from scipy.optimize import minimize

fontsize = 25
fontProperties = {'family':'sans-serif', 'size' : fontsize}
plt.rc('text', usetex=True)   
plt.rc('font', family='sans-serif')  
plt.rc('axes', labelsize = fontsize+5)     # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize-3)    # legend fontsize

mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724

def filter_signal(signal, cutoff_frequency, double_mirror = True):
    
    right_attach = 2*signal[-1]-signal[::-1]
    if double_mirror:
        left_attach = 2*signal[0]-signal[::-1]
        continued_signal = np.hstack((left_attach, signal, right_attach))
    else:
        continued_signal = np.hstack((signal, right_attach))

    windows = sci.firwin(numtaps = continued_signal.shape[0]//10, cutoff = cutoff_frequency,\
                         window='hamming', fs = 500)
    # example_filtered = sci.filtfilt(b = windows, a = [1], x = continued_signal)
    example_filtered = sci.filtfilt(b = windows, a = [1], x = continued_signal, padlen = 7, padtype = 'even')
    clip_index = signal.shape[0]
    if double_mirror:
        clipped_signal = example_filtered[clip_index:2*clip_index]
    else:
        clipped_signal = example_filtered[:clip_index]
    return clipped_signal

def prepare_data(case, fluid, run, cutoff_frequency):
    fol = 'C:\Anna\Fall' + os.sep +  fluid + os.sep + case + os.sep + run + os.sep + 'data_files'
    prefix = case + '_' + run + '_'
    ca_cosh = np.genfromtxt(fol + os.sep + prefix + 'ca_cosh.txt')
    ca_gauss = np.genfromtxt(fol + os.sep + prefix + 'ca_gauss.txt')
    h = np.genfromtxt(fol + os.sep + prefix + 'h_avg.txt')
    h_cl = np.genfromtxt(fol + os.sep + prefix + 'h_cl_r.txt')
    # plt.figure()
    # plt.plot(h)
    vel_raw = np.gradient(h, 0.002)
    vel_cl_raw = np.gradient(h_cl, 0.002)
    
    vel_clean = filter_signal(vel_raw, cutoff_frequency, double_mirror = False)
    vel_clean[np.abs(vel_clean)<4] = 0
    vel_cl_clean = filter_signal(vel_cl_raw, cutoff_frequency, double_mirror = False)
    vel_cl_clean[np.abs(vel_cl_clean)<4] = 0    
    
    ca_clean = filter_signal(ca_gauss, cutoff_frequency, double_mirror = False)
    
    valid_idx = np.argmax(np.abs(vel_clean) > 1)
    vel_clean = vel_clean[valid_idx:]
    vel_cl_clean = vel_cl_clean[valid_idx:]
    ca_clean = ca_clean[valid_idx:]
    vel_raw = vel_raw[valid_idx:]
    ca_gauss = ca_gauss[valid_idx:]
    ca_cosh = ca_cosh[valid_idx:]
    
    vel_raw = vel_raw * mu_w / sigma_w
    vel_clean = vel_clean* mu_w / sigma_w
    
    return ca_gauss, ca_cosh, ca_clean, vel_raw/1000, vel_clean/1000

Speeds = np.array(['fast', 'middle', 'slow']) 

ca_gauss_A_f, ca_cosh_A_f, ca_A_f, v_r_A_f, v_A_f = prepare_data('fast','Water','A', 6)
ca_gauss_B_f, ca_cosh_B_f, ca_B_f, v_r_B_f, v_B_f = prepare_data('fast','Water','B', 6)
ca_gauss_C_f, ca_cosh_C_f, ca_C_f, v_r_C_f, v_C_f = prepare_data('fast','Water','C', 6)

ca_gauss_A_m, ca_cosh_A_m, ca_A_m, v_r_A_m, v_A_m = prepare_data('middle','Water','A', 6)
ca_gauss_B_m, ca_cosh_B_m, ca_B_m, v_r_B_m, v_B_m = prepare_data('middle','Water','B', 6)
ca_gauss_C_m, ca_cosh_C_m, ca_C_m, v_r_C_m, v_C_m = prepare_data('middle','Water','C', 6)

ca_gauss_A_s, ca_cosh_A_s, ca_A_s, v_r_A_s, v_A_s = prepare_data('slow','Water','A', 6)
ca_gauss_B_s, ca_cosh_B_s, ca_B_s, v_r_B_s, v_B_s = prepare_data('slow','Water','B', 6)
ca_gauss_C_s, ca_cosh_C_s, ca_C_s, v_r_C_s, v_C_s = prepare_data('slow','Water','C', 6)

# h_test = np.genfromtxt('C:\Anna\Fall\Water\middle\A\data_files\middle_A_h_avg.txt')
# ca_test = np.genfromtxt('C:\Anna\Fall\Water\middle\A\data_files\middle_A_ca_gauss.txt')
# ca_c_test = np.genfromtxt('C:\Anna\Fall\Water\middle\A\data_files\middle_A_ca_cosh.txt')

# fig = plt.figure()
# ax = plt.gca()
# ax.scatter(v_A_f, ca_gauss_A_f, s = 3)
# ax.scatter(v_B_f, ca_gauss_B_f, s = 3)
# ax.scatter(v_C_f, ca_gauss_C_f, s = 3)
# ax.set_xlabel('$Ca$')
# ax.set_ylabel('$\Theta_{gauss}$')
#%%
fig = plt.figure(figsize = (8, 5))
ax = plt.gca()
ax.scatter(v_A_f, ca_A_f, s = 3, label = 'Fast')
ax.scatter(v_B_f, ca_B_f, s = 3, label = 'Fast')
ax.scatter(v_C_f, ca_C_f, s = 3, label = 'Fast')
ax.scatter(v_A_m, ca_A_m, s = 3, label = 'Middle')
ax.scatter(v_B_m, ca_B_m, s = 3, label = 'Middle')
ax.scatter(v_C_m, ca_C_m, s = 3, label = 'Middle')
ax.grid(b = True)
# ax.scatter(v_A_s, ca_cosh_A_s, s = 3)
# ax.scatter(v_B_s, ca_B_s, s = 3)
# ax.scatter(v_C_s, ca_cosh_C_s, s = 3)
ax.set_xticks(np.linspace(-0.003, 0, 4, dtype = np.float32))
ax.set_xlim(-0.0030005, 0)
ax.set_xlabel('$Ca$')
ax.set_ylabel('$\Theta$[deg]')
fig.tight_layout()
ax.legend(ncol = 2, loc = 'upper left', fontsize = 15)
fig.savefig('velocity_fall.png', dpi = 300)
#%%

fig = plt.figure(figsize = (8, 6))
ax = plt.gca()
ax.scatter(np.gradient(v_A_f*sigma_w/mu_w, 0.002), ca_A_f, s = 3, label = 'Fast', color = 'tab:blue')
ax.scatter(np.gradient(v_A_m*sigma_w/mu_w, 0.002), ca_A_m, s = 3, label = 'Middle', color = 'tab:red')
ax.scatter(np.gradient(v_B_f*sigma_w/mu_w, 0.002), ca_B_f, s = 3, label = 'Fast', color = 'tab:orange')
ax.scatter(np.gradient(v_B_m*sigma_w/mu_w, 0.002), ca_B_m, s = 3, label = 'Middle', color = 'tab:purple')
ax.scatter(np.gradient(v_C_f*sigma_w/mu_w, 0.002), ca_C_f, s = 3, label = 'Fast', color = 'tab:green')
ax.scatter(np.gradient(v_C_m*sigma_w/mu_w, 0.002), ca_C_m, s = 3, label = 'Middle', color = 'tab:brown')
# ax.scatter(v_A_s, ca_cosh_A_s, s = 3)
# ax.scatter(v_B_s, ca_B_s, s = 3)
# ax.scatter(v_C_s, ca_cosh_C_s, s = 3)
ax.legend(ncol = 3, loc = 'center', bbox_to_anchor = (0.5, 1.1), fontsize = 15)
ax.set_xlabel('$a$[m/s$^2$]')
ax.set_ylabel('$\Theta$[deg]')
ax.grid(b = True)
fig.tight_layout()
fig.savefig('acceleration_fall.png', dpi = 300)

# # plt.figure()
# # plt.plot(ca_A_m)
# # plt.plot(ca_B_m)
# # plt.plot(ca_C_m)

# plt.figure()
# plt.plot(v_r_A_s)
# plt.plot(v_r_B_s)
# plt.plot(v_r_C_s)
#%%
# plt.figure()
# plt.plot(ca_A_f)
# plt.plot(ca_B_f)
# plt.plot(ca_C_f)

# plt.figure()
# plt.plot(v_A_s)
# plt.plot(v_B_s)
# plt.plot(v_C_s)

# plt.plot(h_test)
# grad = np.gradient(h_test, 0.002)[-213:]*mu_w/sigma_w
# plt.plot(grad)
# plt.plot(v_A_m)
# plt.plot(ca_test[-213:])
# plt.plot(ca_A_m)
# plt.plot(ca_c_test[-213:])

#%%

def func(x, a, b, c):
    return a*np.power(x, b) + c

def cost(k, theta_exp, v_exp):
    theta_pred = func(v_exp, k[0], k[1], k[2])
    return np.linalg.norm(theta_exp-theta_pred)
    
# x0 = np.array([200,0.5, 1])
# sol = minimize(cost, x0 = x0, args = (ca_cosh_A, v_R_A), method = 'Powell')

# theta_fit = func(v_R_A, sol.x[0], sol.x[1], sol.x[2])
# plt.figure()
# # plt.plot(v_R_A, theta_fit, c = 'r')
# plt.scatter(v_R_A, ca_cosh_A, s = 3)
