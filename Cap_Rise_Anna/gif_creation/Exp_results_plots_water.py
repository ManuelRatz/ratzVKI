# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:11:56 2020
@author: fiorini
@description:
"""

import sys
sys.path.append('../libraries')
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')
# now, to clear the screen
cls()

import numpy as np
import math
#from smoothn import smoothn

import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)
from smoothn import smoothn
from scipy.signal import savgol_filter


def cls():
    os.system('cls' if os.name=='nt' else 'clear')
cls()
plt.close("all")

# =============================================================================
# Input parameters
# =============================================================================


folder = 'D:/Domenico/2DchannelFlow/Experiments/Anna/20200825/1500Pa_up/test1/'
filename  = folder+'2020-07-28-700Pa-pressure1.txt'  # path of the pressure signal
filename_interp  = folder+'2020-07-28-700Pa-pressure_interp.txt'  # path of the pressure signal


# =============================================================================
# functions
# =============================================================================
# plotting settings
SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
         

    
# =============================================================================
# pressure correction
# =============================================================================
# substitute the initial data of pressure (spike) with a linear interpolation
start_index = 0
end_index   = 40

# =============================================================================
# initial conditions
# =============================================================================
t0   = 0;
tend = 1.5;
h0   = 0.;
dh0  = 0.;
no   = 1500
t    = np.linspace(t0,tend,no);

# =============================================================================
# Physical properties
# =============================================================================
# Atmosferic variables
g      = 9.81;         # Gravity (m/s2)
Patm   = 1e5 # atmospheric pressure, Pa


# Water
rhoL   = 1000;         #Fluid Density (kg/m3)
mu     = 8.90*10**(-4); # Fuid dynamic viscosity (Pa*s)
sigma  = 0.07197;      # surface tension (N/m)
thetaS = math.radians(70); # Equilibrium contact angle on quartz



# =============================================================================
# Import pressure signal
# =============================================================================
a = 208
b = -10

def read_lvm(path):
    header    = 12
    value     = 10
    with open(path) as alldata:
        line = alldata.readlines()[14]
    n_samples = int(line.strip().split('\t')[1])
    
    t_p = []
    v_p = []
    for i in range(value):
        with open(path) as alldata:                       #read the data points with the context manager
            lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
        t_p_temp        = [float(line.strip().split('\t')[0]) for line in lines] 
        v_p_temp        = [float(line.strip().split('\t')[1]) for line in lines]
        t_p             = np.append(t_p,t_p_temp)
        v_p             = np.append(v_p,v_p_temp)
        p               = a*v_p+b
        
    return t_p, p

# path900 = 'D:/Domenico/2DchannelFlow/Experiments/Anna/Test20200804/test_wl24mm/test1_900Pa/test-pressure_001.lvm'
# t_p_900,p_900 = read_lvm(path900)

path1500 = 'D:/Domenico/2DchannelFlow/Experiments/Anna/WATER/20200825/1250Pa_up/test1/test-HS-pressure_001.lvm'
t_p_1500,p_1500 = read_lvm(path1500)


# =============================================================================
# Import contact angle and height
# =============================================================================

# path900_RCA = 'D:/Domenico/2DchannelFlow/Experiments/Anna/Test20200804/test_wl24mm/test1_900Pa/2020-08-04-900Pa-RCA.txt'
path1500_results = 'D:/Domenico/2DchannelFlow/Experiments/Anna/WATER/20200825/1500Pa_up/test1/2020-08-16-350Pa_C0-AllResults.txt'
path1250_results = 'D:/Domenico/2DchannelFlow/Experiments/Anna/WATER/20200825/1250Pa_up/test1/2020-08-25-1250Pa-AllResults.txt'
path1000_results = 'D:/Domenico/2DchannelFlow/Experiments/Anna/WATER/20200825/1000Pa_up/test1/2020-08-25-1000Pa-AllResults.txt'

def readResultsFile(path):
    a = 208
    b = -10
    with open(path) as alldata:                       #read the data points with the context manager
        lines = alldata.readlines()[2:]
    t_a             = [float(line.strip().split()[0]) for line in lines] 
    angle_r         = [float(line.strip().split()[4]) for line in lines]
    h               = [float(line.strip().split()[5]) for line in lines]
    p               = [float(line.strip().split()[6]) for line in lines]
    
    t_exp = np.array(t_a)
    RCA = np.degrees(np.array(angle_r))
    h_cl = np.array(h)
    pressure               = a* np.array(p)+b
    return t_exp, RCA, h_cl, pressure

#t_a_900, RCA_900, h_cl_900 = readCAfile(path900_RCA)
t_a_1500, RCA_1500, h_cl_1500, pressure_1500 = readResultsFile(path1500_results)
t_a_1250, RCA_1250, h_cl_1250, pressure_1250 = readResultsFile(path1250_results)
t_a_1000, RCA_1000, h_cl_1000, pressure_1000 = readResultsFile(path1000_results)



# =============================================================================
# Plot pressure
# =============================================================================

plt.figure()
#plt.plot(t_p_900,p_900, 'x-',color='darkblue',linewidth=3,label = '900 Pa')
plt.plot(t_a_1500,pressure_1500, '-',color='darkorange',linewidth=2, label ='1500 Pa')
plt.plot(t_a_1250,savgol_filter(pressure_1250, 101, 3, axis =0), '-',color='darkblue',linewidth=2, label ='1250 Pa')
plt.plot(t_a_1000,savgol_filter(pressure_1000, 101, 3, axis =0), '-',color='green',linewidth=2, label ='1000 Pa')
plt.xlabel('time [s]')
plt.ylabel('Box pressure [Pa]')
plt.xlim([0,2])
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('WATER/Pressure1500.pdf')


h_cl_1500[np.where(h_cl_1500 == 0)[0]]= np.nan
h_cl_1250[np.where(h_cl_1250 == 0)[0]]= np.nan
h_cl_1000[np.where(h_cl_1000 == 0)[0]]= np.nan

RCA_1500 = RCA_1500[~np.isnan(h_cl_1500)]
RCA_1500s = savgol_filter(RCA_1500, 21, 3, axis =0)
t_a_1500 = t_a_1500[~np.isnan(h_cl_1500)]
h_cl_1500 = h_cl_1500[~np.isnan(h_cl_1500)]

RCA_1250 = RCA_1250[~np.isnan(h_cl_1250)]
RCA_1250s = savgol_filter(RCA_1250, 21, 3, axis =0)
t_a_1250 = t_a_1250[~np.isnan(h_cl_1250)]
h_cl_1250 = h_cl_1250[~np.isnan(h_cl_1250)]

RCA_1000 = RCA_1000[~np.isnan(h_cl_1000)]
RCA_1000s = savgol_filter(RCA_1000, 21, 3, axis =0)
t_a_1000 = t_a_1000[~np.isnan(h_cl_1000)]
h_cl_1000 = h_cl_1000[~np.isnan(h_cl_1000)]



h_cl_1500_s =  savgol_filter(h_cl_1500, 121, 3, axis =0)
v_cl_1500 = np.gradient(h_cl_1500,t_a_1500)
v_cl_1500s = np.gradient(h_cl_1500_s,t_a_1500)
v_cl_1500ss = savgol_filter(np.gradient(h_cl_1500_s,t_a_1500), 101, 3, axis =0)
a_cl_1500 = np.gradient(v_cl_1500ss,t_a_1500)

h_cl_1250_s =  savgol_filter(h_cl_1250, 121, 3, axis =0)
v_cl_1250 = np.gradient(h_cl_1250,t_a_1250)
v_cl_1250s = np.gradient(h_cl_1250_s,t_a_1250)
v_cl_1250ss = savgol_filter(np.gradient(h_cl_1250_s,t_a_1250), 101, 3, axis =0)
a_cl_1250 = np.gradient(v_cl_1250ss,t_a_1250)

h_cl_1000_s =  savgol_filter(h_cl_1000, 121, 3, axis =0)
v_cl_1000 = np.gradient(h_cl_1000,t_a_1000)
v_cl_1000s = np.gradient(h_cl_1000_s,t_a_1000)
v_cl_1000ss = savgol_filter(np.gradient(h_cl_1000_s,t_a_1000), 101, 3, axis =0)
a_cl_1000 = np.gradient(v_cl_1000ss,t_a_1000)


# =============================================================================
# Plot height
# =============================================================================

plt.figure()
plt.plot(t_a_1500,h_cl_1500_s*1000+74, '-',color='darkorange',linewidth=3, label ='1500 Pa')
#plt.plot(t_a_1250,h_cl_1250_s*1000+54, '-',color='darkblue',linewidth=2, label ='1250 Pa')
#plt.plot(t_a_1000,h_cl_1000_s*1000+32, '-',color='green',linewidth=2, label ='1000 Pa')
plt.xlabel('time [s]')
plt.ylabel('$h(t)$ [mm]')
plt.xlim([0,2])
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('WATER/Displacement1500.pdf')

# =============================================================================
# Plot fft
# =============================================================================
fs=500
Signal_FFT = np.fft.fft(h_cl_1500)/np.sqrt(len(h_cl_1500)) # Compute the DFT
Freqs=np.fft.fftfreq(len(h_cl_1500))*fs
plt.figure()
plt.plot(Freqs,np.abs(Signal_FFT), '-',color='darkorange',linewidth=3, label ='1500 Pa')
#plt.plot(t_a_1250,h_cl_1250_s*1000+54, '-',color='darkblue',linewidth=2, label ='1250 Pa')
#plt.plot(t_a_1000,h_cl_1000_s*1000+32, '-',color='green',linewidth=2, label ='1000 Pa')
plt.xlabel('freq [1/s]')
plt.ylabel('$h(t)$ [mm]')
plt.xlim([0,2])
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('WATER/Displacement1500.pdf')

# =============================================================================
# Plot velocity
# =============================================================================

plt.figure()

plt.plot(t_a_1500,v_cl_1500s*1000, '-',color='darkorange',linewidth=2, label ='1500 Pa')
plt.plot(t_a_1250,v_cl_1250s*1000, '-',color='darkblue',linewidth=2, label ='1250 Pa')
plt.plot(t_a_1000,v_cl_1000s*1000, '-',color='green',linewidth=2, label ='1000 Pa')

plt.xlabel('time [s]')
plt.ylabel('$v(t)$ [mm/s]')
plt.xlim([0,2])
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('WATER/Velocity.pdf')

# =============================================================================
# Plot capillary number
# =============================================================================

plt.figure()
#plt.plot(t_a_900,v_cl_900s*mu/sigma, 'x-',color='darkblue',linewidth=3,label = 'l')
plt.plot(t_a_1500,v_cl_1500s*mu/sigma, 'x-',color='darkorange',linewidth=2, label ='')
plt.xlabel('time [s]')
plt.ylabel('Ca [-]')
plt.xlim([0,2])
plt.grid()
plt.legend()
plt.tight_layout()


# =============================================================================
# Plot acceleration
# =============================================================================

plt.figure()
plt.plot(t_a_1500,a_cl_1500*1000, 'x',color='darkorange',linewidth=2, label ='1500 Pa')
plt.plot(t_a_1250,a_cl_1250*1000, 'x',color='darkblue',linewidth=2, label ='1250 Pa')
plt.plot(t_a_1000,a_cl_1000*1000, 'x',color='green',linewidth=2, label ='1000 Pa')
#plt.plot(t_a_1500,v_cl_1500*1000, '-',color='darkblue',linewidth=2, label ='')
plt.xlabel('time [s]')
plt.ylabel('$a(t) [m^2/s]$')
plt.xlim([0,2])
plt.grid()
plt.tight_layout()
plt.legend()
#plt.savefig('WATER/Acceleration.pdf')

# =============================================================================
# Plot contact angle
# =============================================================================

plt.figure()
#plt.plot(t_a_900,v_cl_900s*1000, 'x-',color='darkblue',linewidth=3,label = 'l')
plt.plot(t_a_1500,RCA_1500s, '-',color='darkorange',linewidth=2, label ='1500 Pa')
plt.plot(t_a_1250,RCA_1250s, '-',color='darkblue',linewidth=2, label ='1250 Pa')
plt.plot(t_a_1000,RCA_1000s, '-',color='green',linewidth=2, label ='1000 Pa')
#plt.plot(t_a_1500,v_cl_1500*1000, '-',color='darkblue',linewidth=2, label ='')
plt.xlabel('time [s]')
plt.ylabel('Contact angle [deg]')
plt.xlim([0,2])
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('WATER/Contact_angle.pdf')

# =============================================================================
# Plot capillary number - contact angle
# =============================================================================

plt.figure()
sctr = plt.scatter(x=v_cl_1500[:1500]*mu/sigma, y=RCA_1500[:1500], c=a_cl_1500[:1500], cmap='winter',label="Experimental result - Positiv acceleration", zorder=3)
cb1 = plt.colorbar(sctr)
cb1.set_label('$a [m/s^2]$', labelpad=-40, y=1.1, rotation=0,fontsize=15)
# sctr2 = plt.scatter(x=v_cl_1250s[:1000]*mu/sigma, y=RCA_1250[:1000], c=a_cl_1250[:1000], cmap='summer',label="Experimental result - Positiv acceleration", zorder=3)
# cb2 = plt.colorbar(sctr2)
# cb2.set_label('$a [m/s^2]$', labelpad=-40, y=1.1, rotation=0,fontsize=15)
# sctr3 = plt.scatter(x=v_cl_1000s[:1000]*mu/sigma, y=RCA_1000[:1000], c=a_cl_1000[:1000], cmap='autumn',label="Experimental result - Positiv acceleration", zorder=3)
# cb3 = plt.colorbar(sctr3)
# cb3.set_label('$a [m/s^2]$', labelpad=-40, y=1.1, rotation=0,fontsize=15)
plt.xlabel('Capillary number [-]')
plt.ylabel('Contact angle [deg]')
# #plt.xlim([-0.002,0.004])
# #plt.xlim([0.000001,0.003])
# #plt.legend()
plt.grid(zorder=0)
plt.tight_layout()
#plt.savefig('WATER/CA_Capillary_1500.pdf')

plt.figure()
sctr = plt.scatter(x=v_cl_1500s[:1500]*mu/sigma, y=RCA_1500[:1500], c=a_cl_1250[:1500], cmap='winter',label="Experimental result - Positiv acceleration", zorder=3)
cb1 = plt.colorbar(sctr)
cb1.set_label('$a [m/s^2]$', labelpad=-40, y=1.1, rotation=0,fontsize=15)
plt.xlabel('Capillary number [-]')
plt.ylabel('Contact angle [deg]')
# #plt.xlim([-0.002,0.004])
# #plt.xlim([0.000001,0.003])
# #plt.legend()
plt.grid(zorder=0)
plt.tight_layout()
#plt.savefig('WATER/CA_Capillary_1250.pdf')

plt.figure()
sctr = plt.scatter(x=v_cl_1500s[:1000]*mu/sigma, y=RCA_1500s[:1000], c=a_cl_1000[:1000], cmap='winter',label="Experimental result - Positiv acceleration", zorder=3)
cb1 = plt.colorbar(sctr)
cb1.set_label('$a [m/s^2]$', labelpad=-40, y=1.1, rotation=0,fontsize=15)
plt.xlabel('Capillary number [-]')
plt.ylabel('Contact angle [deg]')
# #plt.xlim([-0.002,0.004])
# #plt.xlim([0.000001,0.003])
# #plt.legend()
plt.grid(zorder=0)
plt.tight_layout()
#plt.savefig('WATER/CA_Capillary_1000.pdf')

