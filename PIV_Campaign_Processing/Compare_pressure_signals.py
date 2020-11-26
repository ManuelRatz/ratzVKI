# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 08:41:50 2020

@author: manue
"""


import numpy as np
import post_processing_functions as ppf
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

fol1 = 'C:\PIV_Campaign\Rise\h2\p1500\R_h2_f1200_1_p15' + os.sep
fol2 = 'C:\PIV_Campaign\Rise\h1\p1500\R_h1_f1200_1_p15' + os.sep
fol3 = 'C:\PIV_Campaign\Rise\h4\p1500\R_h4_f1200_1_p15' + os.sep

Fol_Pres_Anna = 'C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\\new_processing\Run_A' + os.sep
pres_Anna = np.genfromtxt(Fol_Pres_Anna + 'pressure_1500.txt')

valve_p_1 = fol1 +'R_h2_f1200_1_p15-utube-valve_p_001.lvm'
pressure_1 = fol1 +'R_h2_f1200_1_p15-utube-pressure_001.lvm'
valve_p_2 = fol2 +'R_h1_f1200_1_p15-utube-valve_p_001.lvm'
pressure_2 = fol2 +'R_h1_f1200_1_p15-utube-pressure_001.lvm'
valve_p_3 = fol3 +'R_h4_f1200_1_p15-utube-valve_p_001.lvm'
pressure_3 = fol3 +'R_h4_f1200_1_p15-utube-pressure_001.lvm'

time_1, valve_1 = ppf.read_lvm(valve_p_1)
f_acq_1 = int((len(time_1)-1)/time_1[-1])
time_1, pres_1 = ppf.read_lvm(pressure_1)

time_2, valve_2 = ppf.read_lvm(valve_p_2)
f_acq_2 = int((len(time_2)-1)/time_2[-1])
time_2, pres_2 = ppf.read_lvm(pressure_2)

time_3, valve_3 = ppf.read_lvm(valve_p_3)
f_acq_3 = int((len(time_3)-1)/time_3[-1])
time_3, pres_3 = ppf.read_lvm(pressure_3)

seconds = 4
start_1 = np.argmax(valve_1 > 500)+220
end_1 = start_1+seconds*f_acq_1
start_2 = np.argmax(valve_2 > 500)
end_2 = start_2+seconds*f_acq_2
start_3 = np.argmax(valve_3 > 500)
end_3 = start_3+seconds*f_acq_3

# pressure_1 = savgol_filter(pres_1[start_1:end_1], 35, 1, axis = 0)
# pressure_2 = savgol_filter(pres_2[start_2:end_2], 23, 1, axis = 0)
# pressure_3 = savgol_filter(pres_3[start_3:end_3], 23, 1, axis = 0)
# t = np.linspace(0,seconds,seconds*f_acq_1)
# fig, ax = plt.subplots()
# # ax.plot(t, valve_1[start:end])
# ax.plot(t, pressure_1, label = 'PIV Pressure')
# t_new = np.linspace(0,4,2002)
# ax.plot(t_new, pres_Anna, label = 'LeDaR Pressure')
# # ax.plot(t, pressure_2)
# # ax.plot(t, pressure_3)
# ax.set_xlim(0,4)
# ax.set_ylim(960,1010)
# ax.legend()
# ax.grid(b=True)

# pres_save = np.vstack((pressure_1, pres_Anna))

np.savetxt('pressure_piv.txt', pres_1[start_1:end_1])
np.savetxt('pressure_ledar.txt', pres_Anna)
