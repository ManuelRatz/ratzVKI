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
from scipy.optimize import curve_fit

fontsize = 20
plt.rc('text', usetex=True)   
plt.rc('font', family='sans-serif')  
plt.rc('axes', labelsize = fontsize+5)     # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize-3)    # legend fontsize

mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724
g = 9.81
# theta_s = 34

def filter_signal(signal, cutoff_frequency, double_mirror = True):
    
    right_attach = 2*signal[-1]-signal[::-1]
    if double_mirror:
        left_attach = 2*signal[0]-signal[::-1]
        continued_signal = np.hstack((left_attach, signal, right_attach))
    else:
        continued_signal = np.hstack((signal, right_attach))

    # continued_signal = signal
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
    cut_end = 1
    ca_cosh = np.genfromtxt(fol + os.sep + prefix + 'ca_cosh.txt')[:-cut_end]
    ca_gauss = np.genfromtxt(fol + os.sep + prefix + 'ca_gauss.txt')[:-cut_end]
    h = np.genfromtxt(fol + os.sep + prefix + 'h_avg.txt')[:-cut_end]
    h_cl = np.genfromtxt(fol + os.sep + prefix + 'h_cl_r.txt')[:-cut_end]
    # plt.figure()
    # plt.plot(h)
    vel_raw = np.gradient(h, 0.002)
    vel_cl_raw = np.gradient(h_cl, 0.002)
    
    vel_clean = filter_signal(vel_raw, cutoff_frequency, double_mirror = False)
    vel_clean[np.abs(vel_clean)<0] = 0
    vel_cl_clean = filter_signal(vel_cl_raw, cutoff_frequency, double_mirror = False)
    vel_cl_clean[np.abs(vel_cl_clean)<0] = 0    
    
    ca_clean = filter_signal(ca_gauss, cutoff_frequency, double_mirror = False)
    
    theta_s = np.mean(ca_cosh[500:])
    # print(theta_s)
    n = 2
    # theta_s = 0
    valid_idx = np.argmax(np.abs(vel_clean) > 1)
    vel_clean = (-1)**n*vel_clean[valid_idx:]
    vel_cl_clean = (-1)**n*vel_cl_clean[valid_idx:]
    ca_clean = (-1)**n*(ca_clean[valid_idx:]-theta_s)
    vel_raw = (-1)**n*vel_raw[valid_idx:]
    ca_gauss = (-1)**n*(ca_gauss[valid_idx:]-theta_s)
    ca_cosh = (-1)**n*(ca_cosh[valid_idx:]-theta_s)
    
    vel_raw = vel_raw * mu_w / sigma_w
    vel_clean = vel_clean* mu_w / sigma_w
    
    return ca_gauss, ca_cosh, ca_clean, vel_raw/1000, vel_clean/1000, theta_s

Theta_gauss, Theta_cosh, Theta, Ca_raw, Ca, Theta_s = prepare_data('fast','Water','C', 6)
G = np.gradient(Ca/mu_w*sigma_w, 0.002)/g


# plt.plot(ca_cosh)
plt.plot(Theta_gauss)
plt.plot(Theta)

plt.figure()
plt.plot(Ca_raw)
plt.plot(Ca)

plt.figure()
plt.plot(G)

# plt.figure()
# plt.scatter(Ca_raw, Theta_gauss, s = 3)
# plt.scatter(Ca, Theta, s = 3)

# plt.figure()
# plt.scatter(Ca, Theta, s = 3)
#%%

# def test_fit(X, a, b, c):
#     x, y = X
#     return a*np.power(x, b)*np.power(y, c)

# def cost(fit_values, x, y, data):
#     data_pred = test_fit((x, y), fit_values[0], fit_values[1], fit_values[2])
#     norm = np.linalg.norm(data-data_pred)
#     return norm

# x_test = np.linspace(0.1, 0.5, 101)
# y_test = np.linspace(0.1, 2, 101)
# data_test = test_fit((x_test, y_test), 2, 1.4, 0.6)+0.05*np.random.rand(x_test.shape[0])

# x0 = np.array([1, 1, 1])
# sol = minimize(cost, x0, args = (x_test, y_test, data_test,))
# solution = sol.x

# data_fit = test_fit((x_test, y_test), solution[0], solution[1], solution[2])
# plt.figure()
# ax = plt.gca()
# ax.scatter(data_test, data_fit, s = 3)
# ax.plot(data_test, data_test, lw = 1, color = 'r')
# ax.set_aspect(1)

# plt.figure()
# ax = plt.gca()
# ax.plot(data_test, label = 'test data')
# ax.plot(data_fit, label = 'fitted data')
# ax.legend()
#%%
"""
This is the fitting function using complex values and casting it to the real
value again
"""
# skip = 0

Ca_cmplx = Ca.astype(np.complex)
G_cmplx = G.astype(np.complex)

def fitting_function_cmplx(X, a, b, c):
    velocity, acceleration = X
    # print(np.power(velocity, b))
    # print(np.power(acceleration, c))
    theta = (a*(velocity**b)*(acceleration**c))
    # print(theta.imag[-1])
    # print(theta.real[-1])
    theta = theta.real
    return theta

def cost_cmplx(fit_values, velocity, acceleration, Theta_exp):
    theta_pred = fitting_function_cmplx((velocity, acceleration), fit_values[0],
                                  fit_values[1], fit_values[2])
    norm = np.linalg.norm(Theta_exp - theta_pred)
    return norm

x0 = np.array([170, 0.5, 0.5])

# bounds = ((-300, 300), (0.00001, np.inf), (0.00001, np.inf))
sol = minimize(cost_cmplx, x0, args = (Ca_cmplx, G_cmplx, Theta,),
               method = 'Nelder-Mead')
print(sol.x)
values = sol.x
theta_pred = fitting_function_cmplx((Ca_cmplx, G_cmplx), values[0], values[1], values[2])
# theta_pred = fitting_function((Ca, G), x0[0], x0[1], x0[2])
# plt.plot(Theta)
# plt.plot(theta_pred)

plt.figure()
ax = plt.gca()
plt.scatter(Theta+Theta_s, theta_pred+Theta_s, s = 3)
ax.set_xlabel('$\Theta_\\textrm{exp}$')
ax.set_ylabel('$\Theta_\\textrm{pred}$')
ax.set_aspect(1)
ax.plot(Theta+Theta_s, Theta+Theta_s, color = 'r')

# plt.figure()
# ax = plt.gca()
# ax.plot(Theta, label = 'Raw')
# ax.plot(theta_pred, label = 'Pred')
# ax.legend(loc = 'lower left')

plt.figure()
plt.scatter(Ca.real, Theta, s = 3, label = 'Raw')
plt.scatter(Ca.real, theta_pred, s = 3, label = 'Pred')
plt.legend(loc = 'upper left')

#%%

def fitting_function(X, a, b, c, d, e):
    velocity, acceleration = X
    theta = (a*(velocity**b)*(acceleration**c))
    return theta