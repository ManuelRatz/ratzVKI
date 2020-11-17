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
folder = 'C:\PIV_Campaign\Rise\h1\p1500\R_h1_f1000_2_p15' + os.sep
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

path_valve_p = folder +'R_h1_f1000_2_p15-utube-valve_p_001.lvm'
path_pressure= folder +'R_h1_f1000_2_p15-utube-pressure_001.lvm'
path_pressure_n= folder +'R_h1_f1000_2_p15-utube-pressure_n_001.lvm'
path_valve_n = folder +'R_h1_f1000_2_p15-utube-valve_n_001.lvm'



#%% main pressure and zero time setting
#pressures
[time, voltages]  = read_lvm(path_pressure)
p_pressure        = [voltage*208.73543056621196-11.817265775905382 for voltage in voltages]

# # valves
[t_valve_p, v_valve_p]   = read_lvm(path_valve_p)
g_valve_p        = np.gradient(v_valve_p,t_valve_p)

f_acq_labview = (len(t_valve_p)-1)/t_valve_p[-1]

start = np.argmax(g_valve_p > 500)
t = np.arange(0, 4, 0.001)
plt.plot(t,p_pressure[start:start+4000])
