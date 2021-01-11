# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:08:38 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter


# plt.plot(np.cos(ca))
for i in range(1, 19, 2):
    folder = 'C:\Anna\Rise\Water\P1250_C30\B\data_files_'+str(i)
    pressure = np.genfromtxt(folder + os.sep + 'P1250_C30_B_pressure.txt')
    h = np.genfromtxt(folder + os.sep + 'P1250_C30_B_h_avg.txt')
    ca = np.genfromtxt(folder + os.sep + 'P1250_C30_B_ca_l.txt')
    t = np.linspace(0, 4, 2000,endpoint = False)
    ca_filter = savgol_filter(ca, 11, 1)
    h_filter = savgol_filter(h, 19, 1)
    vel = np.gradient(h)
    vel2 = np.gradient(h_filter)
    # plt.plot(t, h)
    # plt.plot(np.cos(ca))
    # plt.plot(t,np.cos(ca_filter/180*np.pi))
    # plt.plot(vel)
    # plt.plot(vel2)
    # plt.figure()
    # plt.plot(h)
    # plt.plot(h_filter)
    
    fig = plt.figure()
    idx1 = 0
    idx2 = idx1+1000
    plt.scatter(vel2[idx1:idx2], ca_filter[idx1:idx2], marker='x', s=(13./fig.dpi)**2)