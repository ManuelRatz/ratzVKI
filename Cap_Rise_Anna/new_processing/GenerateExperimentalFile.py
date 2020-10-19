# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:17:20 2020
@author: fiorini
@description: This code change the format of the results with the same header what Domenico uses
"""
import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
# now, to clear the screen
cls()
import sys
sys.path.append('../../libraries')

import numpy as np


#%% inputs
folder = 'test_images' + os.sep + 'raw_images' + os.sep
output_file = 'test.txt'

# Variables changing from experiment to another
T         = 273.+25. # temperature 
f_acq     = 500
h_final   = 74   # final height measured after the test in mm (it is in the TestMatrix.txt file)

# Fluid
name      = 'WATER'  # Available fluids: HFE7000, HFE7100, HFE7200, HFE7300, HFE7500, WATER (everything at Patm),LN2SATURO

thetaS    = np.radians(1e-5);     # Equilibrium contact angle
if name == 'WATER': thetaS    = np.radians(70);     

# Setup dimensions
f         = 0;       # frequency of imposed pressure (Hz) (set to 0 to have a step response)
R         = 0.0025;   # Tube radius/side (m)
W         = R*2      # Channel lateral width (must be equal to the diameter if circular tube)
H         = 0.15
Du        = 3*(2*R)

#%% functions

   
def read_lvm(path):

#path_cam = folder + 'test-THS-cam_001.lvm'
    header    = 12  # number of header rows
    value     = 15  # gives the number of the loop
    with open(path) as alldata:
        line = alldata.readlines()[14]
    n_samples = int(line.strip().split('\t')[1])

    time = []
    voltages = []
    for i in range(value):
        with open(path) as alldata:                       #read the data points with the context manager
            lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
        t_temp        = [float(line.strip().split('\t')[0]) for line in lines] 
        v_temp        = [float(line.strip().split('\t')[1]) for line in lines]
        time          = np.append(time,t_temp)
        voltages      = np.append(voltages,v_temp)
    
    return time,voltages
    
    # paths
path_cam = folder + 'test-HS-cam_001.lvm'
t_cam,v_cam = read_lvm(path_cam)
path_pr = folder + 'test-HS-pressure_001.lvm'
[time, voltages] = read_lvm(path_pr)
p_pressure        = [voltage*208.73543056621196-11.817265775905382 for voltage in voltages]
path_valve = folder + 'test-HS-valve_001.lvm'
t_valve,v_valve = read_lvm(path_valve)


#f_acq = int(len(t_cam)/t_cam[-1])
idx_start_cam = np.argwhere(v_cam > 3)[0]
idx_start_valve = np.argwhere(v_valve > 4.5)[0]
pressure_signal = p_pressure[int(idx_start_valve):]
frame0 = int((t_valve[idx_start_valve]-t_cam[idx_start_cam])*f_acq)

#%% paths and folders (normally they are in agree with the codes in the same folder)


input_height = folder + 'Txts_advanced_fitting/Displacement.txt'
input_height_sx = folder + 'Txts_advanced_fitting/Displacement_CLsx.txt'
input_height_dx = folder + 'Txts_advanced_fitting/Displacement_CLdx.txt'
input_DCAsx = folder + 'Txts_advanced_fitting/LCA.txt'
input_DCAdx = folder + 'Txts_advanced_fitting/RCA.txt'



#%% Loading of fluid properties, interpolation and set of initial condition

#[rho, rhoG, mu, muG, sigma]   = expp.properties(path,T);
if name == 'WATER':
    rho = 997.05
    mu = 0.0008891
    sigma = 0.07197
elif name == 'HFE7200':
    rho = 1430
    mu = 0.00061
    sigma = 0.0136

h_raw = np.loadtxt(input_height)
h_eq = np.mean(h_raw[-100:-1]) # shifting the height
h_raw_sx = np.loadtxt(input_height_sx)
h_raw_dx = np.loadtxt(input_height_dx)
h_exp = (h_raw+(h_final))/1000   # output in meters
h_exp_sx = (h_raw_sx+(h_final))/1000
h_exp_dx = (h_raw_dx+(h_final))/1000
t_exp = time


DCAsx = np.loadtxt(input_DCAsx)
DCAdx = np.loadtxt(input_DCAdx)

DCAsi = np.mean([DCAsx,DCAdx], axis = 0)
thetaS= np.mean(DCAsi[-50:])

tVaA = float(t_valve[idx_start_valve]- t_cam[idx_start_cam])

h = h_exp[frame0:]
h_sx = h_exp_sx[frame0:]
h_dx = h_exp_dx[frame0:]
# lca = np.degrees(DCAsx[frame0:])
# rca = np.degrees(DCAdx[frame0:])
lca = DCAsx[frame0:]*180/np.pi
rca = DCAdx[frame0:]*180/np.pi

t = np.linspace(0,len(h)/f_acq,(len(h)))

Dp        = np.mean(pressure_signal[250:]) # up
# Dp        = np.mean(pressure_signal[0:250]) # down

def saveTxt(path,t_exp,h_exp,h_exp_sx,h_exp_dx,DCAsx,DCAdx,pressure): 
    num_format = '{:20.16f}'
    geo_format = '{:=7.4f}'
    time_format= '{:7.3f}'
    # save txt file of experiment
    header = """------------------Properties-----------------------------------------------------------------------------------------------------------------------------
fluid:        """ + name + """
rho           """ + num_format.format(rho) + ' (kg/m3)' +"""
mu            """ + num_format.format(mu) + ' (Pa s)' +"""
sigma         """ + num_format.format(sigma) + ' (N/m)' +"""
thetaS        """ + num_format.format(thetaS) + ' (deg)' +"""\n
setup:
tube radius   """ + geo_format.format(R) + ' (m)' +"""
tube width    """ + geo_format.format(W) + ' (m)' +"""
tube height   """ + geo_format.format(H) + ' (m)' +"""
liquid volume """ '-' """
tube distance """ '-' """
pressure step """ + geo_format.format(Dp) + ' (Pa)' +"""
frequency     """ + geo_format.format(0) + ' (Hz)' +"""\n
timing:
valve A open  """ + geo_format.format(tVaA) + ' (s)' +"""
valve B open  """ '-' """
------------------Experiment-----------------------------------------------------------------------------------------------------------------------------
  time (s)        height (m)           CL height sx(m)      CL height dx(m)     D.C.A.sx (deg)       D.C.A.sx (deg)        pressure (Pa)
"""
    with open(path,"w+") as thefile:
        thefile.write(header) # write the header
        for i in range(len(h_exp)): # write the data
            thefile.write(time_format.format(t_exp[i]) + ' ' + num_format.format(h_exp[i]) + ' ' + num_format.format(h_exp_sx[i]) + ' ' + num_format.format(h_exp_dx[i])  + ' ' + num_format.format(DCAsx[i]) + ' ' + num_format.format(DCAdx[i]) + ' ' + num_format.format(pressure[i])+'\n')


saveTxt(output_file,t,h,h_sx,h_dx,lca,rca,pressure_signal)