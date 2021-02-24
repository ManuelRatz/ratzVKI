# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:34:51 2021

@author: Manuel Ratz
@description: Test Case for the DESCENDING interface to check the possiblity
    of a correlation from the acceleration and the velocity (both dimensionless)
    One test case is loaded and smoothed, then two fittings are being done:
        One using complex numbers and casting it to the real value
        One using the absolute value and the sign separately
    This is required because we are raising negaitve numbers to a fraction power
"""

import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as sci 
from scipy.optimize import minimize

Fol_Out = 'C:\\Uni_und_wichtiges\VKI\\report_images\plots_lif'
# these are the constants for water
mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724

mu_h = 0.0006777
rho_h = 1429.420
sigma_h = 0.0139

g = 9.81

"""This is a function to filter the raw signals, because the contact angle is
very noisy (see the plot later on). We use the contact angle from the gaussian
process because it better represents the small contact angles."""
def filter_signal(signal, cutoff_frequency):
    # we extend the right side
    # right_attach = 2*signal[-1]-signal[::-1]
    # continued_signal = np.hstack((signal, right_attach))
    # set up the windows for filtfilt
    windows = sci.firwin(numtaps = signal.shape[0]//10, cutoff = cutoff_frequency,\
                         window='hamming', fs = 500)
    # filter the signal
    example_filtered = sci.filtfilt(b = windows, a = [1], x = signal,
                                    padlen = 1000, padtype = 'odd')
    # get the index at which to clip again
    return example_filtered

"""This function prepares the data after giving a test case. Here the smoothing
and velocity calculation is all taken care of. Velocities below a certain threshold
are killed, to eliminate the static stuff from affecting the correlation"""
def prepare_data(case, fluid, run, cutoff_frequency):
    # input folder
    fol = 'data_files' + os.sep + fluid
    # case prefix
    prefix = case + '_' + run + '_'
    # amount of points to cut at the end
    """ This parameter is very important, I will refer to it as cut_ending
    in my descriptions. Changing this one has a large impact on the correlation"""
    cut_end = 1
    # load the files
    ca_cosh = np.genfromtxt(fol + os.sep + prefix + 'ca_cosh.txt')[:-cut_end]
    ca_gauss = np.genfromtxt(fol + os.sep + prefix + 'ca_gauss.txt')[:-cut_end]
    h = np.genfromtxt(fol + os.sep + prefix + 'h_avg.txt')[:-cut_end]
    h_cl = np.genfromtxt(fol + os.sep + prefix + 'h_cl_r.txt')[:-cut_end]
    
    # calculate the velocities
    vel_raw = np.gradient(h_cl, 0.002)
    vel_cl_raw = np.gradient(h_cl, 0.002)
    # filter the velocities
    vel_clean = filter_signal(vel_raw, cutoff_frequency)
    vel_clean[np.abs(vel_clean)<0] = 0
    # vel_clean = vel_raw
    vel_cl_clean = filter_signal(vel_cl_raw, cutoff_frequency)
    vel_cl_clean[np.abs(vel_cl_clean)<0] = 0    
    # vel_cl_clean = vel_cl_clean
    
    # calculate the static contact angle
    theta_s = np.mean(ca_cosh[:500])
    # print(theta_s)
    theta_s = 0
    
    # clip the signals to only have velocities > 3 mm/s
    valid_idx = np.argmax(np.abs(vel_clean) > 3)
    
    if fluid == 'water':
        valid_idx = 100
        last = -800
        mu = mu_w
        sigma = sigma_w
        # filter the contact angle
        ca_clean = filter_signal(ca_gauss, 10)
    elif fluid == 'HFE':
        valid_idx = 0
        last = -800
        mu = mu_h
        sigma = sigma_h
        # filter the contact angle
        ca_clean = filter_signal(ca_gauss, cutoff_frequency)
    else:
        raise ValueError('Invalid Fluid')    

    vel_clean = vel_clean[valid_idx:last]
    vel_cl_clean = vel_cl_clean[valid_idx:last]
    ca_clean = ca_clean[valid_idx:last]-theta_s
    vel_raw = vel_raw[valid_idx:last]
    h = h[valid_idx:last]
    ca_gauss = ca_gauss[valid_idx:last]-theta_s
    ca_cosh = ca_cosh[valid_idx:last]-theta_s

    
    # transform velocities to capillary numbers
    vel_raw = vel_raw * mu / sigma
    vel_clean = vel_clean* mu / sigma
    
    # return the data, the velocity is divided by 1000 to go from mm to m
    return ca_gauss, ca_cosh, ca_clean, vel_raw/1000, vel_clean/1000, theta_s, h

# function for the fitting
def fitting_function(X, a, b, c, theta_static):
    # get velocity and acceleration
    velocity, acceleration = X
    # calculate theta
    theta = a*np.sign(velocity)*np.power(np.abs(velocity), b) + c*acceleration\
        + theta_static
    # return it
    return theta

# cost function for the minimzation
def cost(fit_values, velocity, acceleration, Theta_exp):
    # predict theta
    theta_pred = fitting_function((velocity, acceleration), fit_values[0],
                                  fit_values[1], fit_values[2], fit_values[3])
    # calculate the norm and return it
    norm = np.linalg.norm(Theta_exp - theta_pred)
    return norm

#%%
Theta_gauss, Theta_cosh, Theta, Ca_raw, Ca, Theta_s, Height =\
    prepare_data('P1000_C30','water','C', 6)
# calculate the dimensionless acceleration
G = np.gradient(Ca/mu_w*sigma_w, 0.002)/g

# x0 = np.array([7.7e3, 1.45, -16, 26]) # init values for HFE
x0 = np.array([7.7e3, 1.45, -16, 33]) # init values for water
bounds = ((0.9*x0[0], 1.1*x0[0]), (0.5*x0[1], 1.5*x0[1]), (1.5*x0[2], 0.5*x0[2]),
          (0.95*x0[3], 1.05*x0[3]))
bounds = ((-1e6, 1e6), (-1e6, 1e6), (-1e6, 1e6), (0.95*x0[3], 1.05*x0[3]))
sol = minimize(cost, x0, args = (Ca, G, Theta,), method = 'SLSQP',
                options={'maxiter' : 100000}, bounds = bounds)
print(sol.x)

# calculate the predicted Theta
values = sol.x
theta_pred = fitting_function((Ca, G), values[0], values[1], values[2], values[3])

fontsize = 25
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')  
plt.rc('axes', labelsize = fontsize+5)     # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize-3)    # legend fontsize
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'

t = np.linspace(100/500, 1200/500, Theta.shape[0], endpoint = False)
# plot the predicted vs original contact angle to compare
fig = plt.figure(figsize = (14, 7))
ax = plt.gca()
# ax.plot(Theta+Theta_s, Theta+Theta_s, color = 'r')
# plt.scatter(Theta+Theta_s, theta_pred+Theta_s, s = 3)
# ax.set_aspect(1)
ax.plot(t, Theta, c = 'k', dashes = (1.5, 1.5), label = 'Experimental')
ax.plot(t, theta_pred, c = 'k', dashes = (10, 3), label = 'Prediction')
ax.set_xlim(t.min(), t.max())
ax.set_xticks(np.linspace(0.25, 2.25, 9))
ax.set_xticklabels(np.linspace(0.25, 2.25, 9, dtype = np.float32))
ax.set_yticks(np.linspace(0, 100, 6))
ax.set_yticklabels(np.linspace(0, 100, 6, dtype = np.int32))
ax.set_ylim(0,100)
ax.grid(b = True)
ax.legend(loc = 'lower right', ncol = 2, handlelength = 1.7)
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$\Theta$[-]')
fig.tight_layout()
Name_Out = Fol_Out + os.sep + 'prediction_water'
fig.savefig(Name_Out, dpi = 200)

#%%
Theta_gauss, Theta_cosh, Theta, Ca_raw, Ca, Theta_s, Height =\
    prepare_data('P1500','HFE','C', 5)
# calculate the dimensionless acceleration
G = np.gradient(Ca/mu_h*sigma_h, 0.002)/g

x0 = np.array([7.7e3, 1.45, -56, 26]) # init values for HFE
bounds = ((0.9*x0[0], 1.1*x0[0]), (0.9*x0[1], 1.1*x0[1]), (1.1*x0[2], 0.9*x0[2]),
          (0.95*x0[3], 1.05*x0[3]))
# bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
sol = minimize(cost, x0, args = (Ca, G, Theta,), method = 'SLSQP',
                options={'maxiter' : 100000}, bounds = bounds)
print(sol.x)

# calculate the predicted Theta
values = sol.x
theta_pred = fitting_function((Ca, G), values[0], values[1], values[2], values[3])

t = np.linspace(0/500, 1500/500, Theta.shape[0], endpoint = False)
fig = plt.figure(figsize= (14, 7))
ax = plt.gca()
ax.plot(t, Theta, c = 'k', dashes = (1.5, 1.5), label = 'Experimental')
ax.plot(t, theta_pred, c = 'k', dashes = (10, 3), label = 'Prediction') 
ax.set_xlim(t.min(), t.max())
ax.set_xticks(np.linspace(0, 3, 7))
ax.set_xticklabels(np.linspace(0, 3, 7, dtype = np.float32))
# ax.set_ylim(15, 50)
ax.set_yticks(np.linspace(20, 60, 5))
ax.set_yticklabels(np.linspace(20, 60, 5, dtype = np.int32))
ax.set_ylabel('$\Theta$[-]')
ax.set_xlabel('$t$[s]')
ax.grid(b = True)
ax.legend(loc = 'upper right', handlelength = 1.8)
fig.tight_layout()
Name_Out = Fol_Out + os.sep + 'prediction_HFE'
fig.savefig(Name_Out, dpi = 200)