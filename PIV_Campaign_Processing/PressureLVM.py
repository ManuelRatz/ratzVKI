# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:58:56 2020

@author: fiorini

"""
import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
# now, to clear the screen
cls()
import sys
sys.path.append('../../libraries')

import numpy as np
from matplotlib import pyplot as plt


#%% inputs

#%% functions....

def read_lvm(path):
    header    = 12
    value     = 15
    with open(path) as alldata:
        line = alldata.readlines()[14]
    n_samples = int(line.strip().split('\t')[1])
    time    = []
    voltage = []
    for i in range(value):
        with open(path) as alldata:                       #read the data points with the context manager
            lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
        time_temp       = [float(line.strip().split('\t')[0].replace(',','.')) for line in lines] 
        voltage_temp    = [float(line.strip().split('\t')[1].replace(',','.')) for line in lines]
        time            = np.append(time,time_temp)
        voltage         = np.append(voltage,voltage_temp)
    return [time, voltage]

#%% paths and folders (normally they are in agree with the codes in the same folder)

fol1 = 'C:\PIV_Campaign\Rise\h2\p1500\R_h2_f1200_1_p15' + os.sep
fol2 = 'C:\PIV_Campaign\Rise\h1\p1500\R_h1_f1200_1_p15' + os.sep
fol3 = 'C:\PIV_Campaign\Rise\h4\p1500\R_h4_f1200_1_p15' + os.sep
path_valve_p_1 = fol1 +'R_h2_f1200_1_p15-utube-valve_p_001.lvm'
path_pressure_1 = fol1 +'R_h2_f1200_1_p15-utube-pressure_001.lvm'
path_valve_p_2 = fol2 +'R_h1_f1200_1_p15-utube-valve_p_001.lvm'
path_pressure_2 = fol2 +'R_h1_f1200_1_p15-utube-pressure_001.lvm'
path_valve_p_3 = fol3 +'R_h4_f1200_1_p15-utube-valve_p_001.lvm'
path_pressure_3 = fol3 +'R_h4_f1200_1_p15-utube-pressure_001.lvm'


#%% main pressure and zero time setting
#pressures
[time, voltages]  = read_lvm(path_pressure_3)
p_pressure        = [voltage*208.73543056621196-11.817265775905382 for voltage in voltages]

# # valves
[t_valve_p, v_valve_p]   = read_lvm(path_valve_p_3)
g_valve_p        = np.gradient(v_valve_p,t_valve_p)

f_acq_labview = (len(t_valve_p)-1)/t_valve_p[-1]

start = np.argmax(v_valve_p > 4)
t = np.arange(0, 4, 0.001)
roi = p_pressure[start:start+4000]
from scipy.signal import savgol_filter
roi = savgol_filter(roi, 15, 3, axis = 0)
plt.plot(t,roi)
