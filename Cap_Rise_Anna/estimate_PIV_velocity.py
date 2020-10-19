#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:34:11 2020

@author: ratz
@description: calculate the velocity range needed to set the PIV framerate
"""
import numpy as np                     # for arrays
import matplotlib.pyplot as plt        # for plotting
import os                              # for data paths
from scipy.signal import savgol_filter # for smoothing

pressure_array = (1000, 1250, 1500) # pressure measurements done

for i in range(0,1):
    Fol_In = 'experimental_data' + os.sep + '%d_pascal' %pressure_array[i] +\
        os.sep + 'avg_height_%d.txt' %pressure_array[i] # input folder
    height = np.genfromtxt(Fol_In) # load height
    vel = np.gradient(height)/0.002 # calculate velocity
    vel_smooth = savgol_filter(vel, 55, 2, axis = 0) # smooth the velocity
    t = np.linspace(0, 4, len(height)) # generate time steps
    
    # find the index of the first peak
    idx = np.argmax(height)
    # shift the velocity and time array
    t = t[idx:]
    vel_smooth = vel_smooth[idx:]
    height = height[idx:]
    
    # get the maximum velocity
    vel_max = np.max(np.abs(vel_smooth))
    #calculate the arithmetic mean of max and min
    v_armean = (vel_max)/2 # as the min velocity is 0
    v_armean_plot = np.zeros(len(vel_smooth)) + v_armean
    v_mean = np.mean(np.abs(vel_smooth))
    v_mean_plot = np.zeros(len(vel_smooth))+v_mean
    
    # plot the result
    fig, ax = plt.subplots() # create figure
    ax.plot(t, vel_smooth, label = '$|u|$')
    ax.plot(t, v_armean_plot, label = '$u_{max,min}$')
    ax.plot(t, v_mean_plot, label = '$u_{mean}$')
    # ax.plot(t, (height-0.074)*5)
    ax.set_xlim(t[0], 2.5) # set xlimits, this is the first 4 peaks
    # ax.set_ylim(0, np.max(vel_smooth)*1.05) # set y limits
    ax.grid() # enable grid
    ax.legend()
    ax.set_title('Absolute velocity from first to fourth peak, %d Pa' %pressure_array[i])