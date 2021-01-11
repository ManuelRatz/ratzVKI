# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:14:42 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\new_processing')

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate           # for setting up interpolation functions
from scipy.integrate import odeint      # for solving the ode
import os
from scipy.signal import savgol_filter

Fol_In = 'C:\Anna\Rise\Water\P1500_C30\A'
Case = 'P1500_C30_A'

def load_txt_files(Fol_In, Case):
    Folder = os.path.join(Fol_In, 'data_files')
    pressure = np.genfromtxt(os.path.join(Folder, Case + '_pressure.txt'))
    h_avg = np.genfromtxt(os.path.join(Folder, Case + '_h_avg.txt'))
    ca = np.genfromtxt(os.path.join(Folder, Case + '_ca_r.txt'))
    fit_gauss = np.genfromtxt(os.path.join(Folder, Case + '_gauss_curvature.txt'))
    fit_exp = np.genfromtxt(os.path.join(Folder, Case + '_cosh_curvature.txt'))
    return pressure, h_avg, ca, fit_gauss, fit_exp

pressure, h_avg, ca, fit_gauss, fit_cosh = load_txt_files(Fol_In, Case)
x = np.linspace(-2.5, 2.5, 1000)

#%%

fit_cosh_smo = savgol_filter(fit_cosh, 7, 1)
fit_gauss_smo = savgol_filter(fit_gauss, 7, 1)

aoi = 600
plt.figure()
plt.plot(fit_cosh[aoi:aoi+100], label = 'Cosh')
plt.plot(fit_gauss[aoi:aoi+100], label = 'Gauss')
plt.legend()

# plt.figure()
# plt.plot(fit_cosh_smo, label = 'Cosh')
# plt.plot(fit_gauss_smo, label = 'Gauss')
# plt.legend()

# plt.figure()
# plt.plot(np.cos(ca*np.pi/180)*2)
# plt.plot(fit_gauss_smo*5)