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
from scipy.optimize import curve_fit

# these are the constants for water
mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724
g = 9.81

"""This is a function to filter the raw signals, because the contact angle is
very noisy (see the plot later on). We use the contact angle from the gaussian
process because it better represents the small contact angles."""
def filter_signal(signal, cutoff_frequency):
    # we extend the right side
    right_attach = 2*signal[-1]-signal[::-1]
    continued_signal = np.hstack((signal, right_attach))
    # set up the windows for filtfilt
    windows = sci.firwin(numtaps = continued_signal.shape[0]//10, cutoff = cutoff_frequency,\
                         window='hamming', fs = 500)
    # filter the signal
    example_filtered = sci.filtfilt(b = windows, a = [1], x = continued_signal,
                                    padlen = 10, padtype = 'even')
    # get the index at which to clip again
    clip_index = signal.shape[0]
    clipped_signal = example_filtered[:clip_index]
    return clipped_signal

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
    cut_end = 10
    # load the files
    ca_cosh = np.genfromtxt(fol + os.sep + prefix + 'ca_cosh.txt')[:-cut_end]
    ca_gauss = np.genfromtxt(fol + os.sep + prefix + 'ca_gauss.txt')[:-cut_end]
    h = np.genfromtxt(fol + os.sep + prefix + 'h_avg.txt')[:-cut_end]
    h_cl = np.genfromtxt(fol + os.sep + prefix + 'h_cl_r.txt')[:-cut_end]
    
    # calculate the velocities
    vel_raw = np.gradient(h, 0.002)
    vel_cl_raw = np.gradient(h_cl, 0.002)
    # filter the velocities
    vel_clean = filter_signal(vel_raw, cutoff_frequency)
    vel_clean[np.abs(vel_clean)<0] = 0
    # vel_clean = vel_raw
    vel_cl_clean = filter_signal(vel_cl_raw, cutoff_frequency)
    vel_cl_clean[np.abs(vel_cl_clean)<0] = 0    
    # vel_cl_clean = vel_cl_clean
    
    # filter the contact angle
    ca_clean = filter_signal(ca_gauss, 8)
    
    # calculate the static contact angle
    theta_s = np.mean(ca_cosh[:500])
    # print(theta_s)
    theta_s = 33
    
    # clip the signals to only have velocities > 3 mm/s
    valid_idx = np.argmax(np.abs(vel_clean) > 3)
    valid_idx = 150
    last = 1250
    vel_clean = vel_clean[valid_idx:last]
    vel_cl_clean = vel_cl_clean[valid_idx:last]
    ca_clean = ca_clean[valid_idx:last]-theta_s
    vel_raw = vel_raw[valid_idx:last]
    h = h[valid_idx:last]
    ca_gauss = ca_gauss[valid_idx:last]-theta_s
    ca_cosh = ca_cosh[valid_idx:last]-theta_s
    
    # transform velocities to capillary numbers
    vel_raw = vel_raw * mu_w / sigma_w
    vel_clean = vel_clean* mu_w / sigma_w
    
    # return the data, the velocity is divided by 1000 to go from mm to m
    return ca_gauss, ca_cosh, ca_clean, vel_raw/1000, vel_clean/1000, theta_s, h

""" Here the test case is loaded, I have put three test cases in the repository,
so you can see how different each of them is. They are accessed by either
'A', 'B' or 'C'"""
Theta_gauss, Theta_cosh, Theta, Ca_raw, Ca, Theta_s, Height =\
    prepare_data('P1000_C30','water','A', 4)
# calculate the dimensionless acceleration
G = np.gradient(Ca/mu_w*sigma_w, 0.002)/g


"""In these plots we can see just how messy the dynamic contact angle is"""
# plot the raw vs filtered signal to compare the smoothing
fig = plt.figure(figsize = (8, 5))
ax = plt.gca()
plt.plot(Theta_gauss)
plt.plot(Theta)
ax.set_xlabel('N')
ax.set_ylabel('$\Theta$')

# fig = plt.figure(figsize = (8, 5))
# ax = plt.gca()
# ax.plot(Height)

fig = plt.figure(figsize = (8, 5))
ax = plt.gca()
plt.plot(Ca_raw)
plt.plot(Ca)
ax.set_xlabel('N')
ax.set_ylabel('$Ca$')

fig = plt.figure(figsize = (8, 5))
ax = plt.gca()
plt.plot(G)
ax.set_xlabel('N')
ax.set_ylabel('$a/g$')

# fig = plt.figure(figsize = (8, 5))
# ax = plt.gca()
# ax.scatter(Ca, Theta, s = 3)
# ax.set_xlim(1.1*np.min(Ca),np.max(Ca))

# fig = plt.figure(figsize = (8, 5))
# ax = plt.gca()
# ax.scatter(G, Theta, s = 3)
# ax.set_xlim(1.1*np.min(G),np.max(G))


"""
Now the thing that I have found to be problematic:
    
For a given test case, say 'C', I can run the code and I get a set of fitting points.
If I now change the cut_ending parameter, the smoothing plots for Theta and Ca
only change by a small amount. However, the correlation parameters change by
quite a bit. I used a cut_ending of 20 and 10 to show this. The only way to
improve this in my opinion is to have a more stable fit, that requires less filtering
and can be trusted more. 

Running test case 'B' again gives completely different fitting parameters, here 
you can change cut_ending again to see how it affects the parameters.

Keep in mind that these are the fast releases, the best ones, by a lot. For the
slow and medium releases the results are much much worse, the fitting often even 
fails. 

Now that I am writing this, I have thought to put one additional case, a slow release
for water, case A. The velocity and acceleration are absolutely weird and strange.
My biggest problem is that I was not present for the experiments, so I don't know
how they were conducted and cannot judge if this is plausible'

I have not tried to look at a Rise so far, because the slow case is prove 
enough for me, that we cannot get a correlation for this type of data.

I am however, as always, ready to be proven wrong by you :D

"""

#%%
"""
This is the fitting function using real values with the absolute value for the 
power calculation.
"""

# function for the fitting
def fitting_function(X, a, b, c):
    # get velocity and acceleration
    velocity, acceleration = X
    # calculate theta
    theta = a*np.sign(velocity)*(np.abs(velocity)**b)\
        *np.sign(acceleration)*(np.abs(acceleration)**c)
    # return it
    return theta

# cost function for the minimzation
def cost(fit_values, velocity, acceleration, Theta_exp):
    # predict theta
    theta_pred = fitting_function((velocity, acceleration), fit_values[0],
                                  fit_values[1], fit_values[2])
    # calculate the norm and return it
    norm = np.linalg.norm(Theta_exp - theta_pred)
    return norm

# initial guess
x0 = np.array([1, 0.33, 4])

# calculate the solution with Nelder Mead
sol = minimize(cost, x0, args = (Ca, G, Theta,), method = 'Nelder-Mead',
               options={'maxiter' : 100000, 'adaptive' : True, 'fatol' : 1e-3,
                        'xatol' : 1e-3})
# print it
print(sol.x)
# calculate the predicted Theta
values = sol.x
theta_pred = fitting_function((Ca, G), values[0], values[1], values[2])

# plot the predicted vs original contact angle to compare
plt.figure(figsize = (8, 5))
ax = plt.gca()
# ax.plot(Theta+Theta_s, Theta+Theta_s, color = 'r')
# plt.scatter(Theta+Theta_s, theta_pred+Theta_s, s = 3)
plt.plot(Theta + Theta_s)
plt.plot(theta_pred + Theta_s)
ax.set_xlabel('$\Theta_\\mathsf{exp}$')
ax.set_ylabel('$\Theta_\\mathsf{pred}$')
# ax.set_aspect(1)

#%%

"""
This is the fitting function using complex values and casting it to the real
value again. IGNORE this for now, it is not important.
"""
# skip = 0

# Ca_cmplx = Ca.astype(np.complex)
# G_cmplx = G.astype(np.complex)

# def fitting_function_cmplx(X, a, b, c):
#     velocity, acceleration = X
#     # print(np.power(velocity, b))
#     # print(np.power(acceleration, c))
#     theta = (a*(velocity**b)*(acceleration**c))
#     # print(theta.imag[-1])
#     # print(theta.real[-1])
#     theta = theta.real
#     return theta

# def cost_cmplx(fit_values, velocity, acceleration, Theta_exp):
#     theta_pred = fitting_function_cmplx((velocity, acceleration), fit_values[0],
#                                   fit_values[1], fit_values[2])
#     norm = np.linalg.norm(Theta_exp - theta_pred)
#     return norm

# x0 = np.array([170, 0.5, 0.5])

# # bounds = ((-300, 300), (0.00001, np.inf), (0.00001, np.inf))
# sol = minimize(cost_cmplx, x0, args = (Ca_cmplx, G_cmplx, Theta,),
#                 method = 'Nelder-Mead')
# print(sol.x)
# values = sol.x
# theta_pred = fitting_function_cmplx((Ca_cmplx, G_cmplx), values[0], values[1], values[2])

# plt.figure()
# ax = plt.gca()
# plt.scatter(Theta+Theta_s, theta_pred+Theta_s, s = 3)
# ax.set_xlabel('$\Theta_\\textrm{exp}$')
# ax.set_ylabel('$\Theta_\\textrm{pred}$')
# ax.set_aspect(1)
# ax.plot(Theta+Theta_s, Theta+Theta_s, color = 'r')