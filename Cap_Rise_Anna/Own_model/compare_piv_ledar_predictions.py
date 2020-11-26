# -*- coding: utf-8 -*-
"""
@author ratz
@description calculate the height predicted by the model from the experimental
pressure and contact angle signal
"""

import numpy as np                      # for array operations
import os                               # for setting up data paths
from scipy.signal import savgol_filter  # for smoothing the data
import matplotlib.pyplot as plt         # for plotting results
from scipy import interpolate           # for setting up interpolation functions
from scipy.integrate import odeint      # for solving the ode


plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=20)   # fontsize of the figure title
plt.rc('text', usetex=True)      # use latex for the text
plt.rc('font', family='serif')   # serif as text font
plt.rc('axes', grid=True)        # enable the grid
plt.rc('savefig', dpi = 100)     # set the dpi for saving figures

# set the constants (this is for water)
g      = 9.81;          # Gravity (m/s2)
Patm   = 1e5            # atmospheric pressure, Pa
rhoL   = 1000;          # Fluid Density (kg/m3)
mu     = 8.90*10**(-4); # Fuid dynamic viscosity (Pa*s)
sigma  = 0.07197;       # surface tension (N/m)
r      = 0.0025         # radius of the tube (m)
delta  = 2* r           # channel width (m)
l      = 100*r          # depth of the channel

# Test cases used for evaluation
pressure_case = np.array([1000, 1250, 1500])
# here the pressure steps as measured by the sensor are given
pressure_step = np.array([662.6799, 823.1328, 990.89])

# set up the for loop
for i in range(2, 3):
    # Update the user about the status
    print('Calculating pressure %d of %d' %((i+1), len(pressure_case)))
    # load the data
    FOLDER1 = 'C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing' + os.sep 
    FOLDER2 = 'C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\\new_processing\Run_A' + os.sep
    ca_load = np.genfromtxt(FOLDER2 + 'rca_%d.txt' %pressure_case[i])
    pres_piv = np.genfromtxt(FOLDER1 + 'pressure_piv.txt')
    pres_ledar = np.genfromtxt(FOLDER1 + 'pressure_ledar.txt')
    h_avg_load = np.genfromtxt(FOLDER2 + 'avg_height_%d.txt' %pressure_case[i])
    
    # smooth the data 
    # h_smoothed = savgol_filter(h_avg_load, 55, 2, axis = 0)
    h_smoothed = h_avg_load +0.074
    # pres_piv = savgol_filter(pres_piv, 23, 1, axis = 0)
    # ca_load = savgol_filter(ca_load, 15, 1, axis = 0)
    # pres_ledar = savgol_filter(pres_ledar, 23, 1, axis = 0)
    
    duration = 4
    t_ledar = np.linspace(0, duration, duration*500+1)
    t_piv = np.arange(0, (len(pres_piv)-0.5)/1000, 1/1000) # this must be the labview f_acq
    
    #set up the interpolation functions

    pres_inter = interpolate.splrep(t_piv, pres_piv)
    def pres_piv_f(t):
        return interpolate.splev(t, pres_inter)
    
    pres_inter_2 = interpolate.splrep(t_ledar, pres_ledar)
    def pres_ledar_f(t):
        return interpolate.splev(t, pres_inter_2)
    
    ca_inter = interpolate.splrep(t_ledar, ca_load)
    def ca(t):
        return interpolate.splev(t, ca_inter)
    
    #create the folder to save the images
    SAVE_IMAGES = '%d_pa_images' %pressure_case[i] + os.sep
    if not os.path.exists(SAVE_IMAGES):
        os.mkdir(SAVE_IMAGES)
    #set the inital values
    h0 = h_smoothed[0]
    velocity = np.gradient(h_smoothed)
    dh0 = velocity[0]/(0.002)
    X0 = np.array([dh0, h0])
    
    #Define the ODE functions
    def ode_ledar(X, t_ledar):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_ledar_f(t_ledar)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t_ledar))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_piv(X, t_call):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_piv_f(t_call)/rhoL\
                                                      + 2.0*(l+delta)*sigma*np.cos(ca(t_call))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    
    #calculate the solutions
    solution_ledar = odeint(ode_ledar, X0, t_ledar)
    solution_piv = odeint(ode_piv, X0, t_piv)

    fig, ax = plt.subplots()
    ax.plot(t_ledar, solution_ledar[:,1]*1000, label = 'LeDaR Prediction')
    ax.plot(t_piv, solution_piv[:,1]*1000, label = 'PIV')
    ax.plot(t_ledar, h_smoothed*1000, label = 'LeDaR Measurement')
    ax.legend(loc = 'lower right')
    ax.set_xlim(0,4)
    ax.set_xticks(np.linspace(0,4,9))
    ax.set_ylim(75,116)
    fig.savefig('LeDaR_PIV_h_comparison.png', dpi = 400)
    
    fig, ax = plt.subplots()
    ax.plot(t_piv, pres_piv_f(t_piv), label = 'PIV Pressure')
    ax.plot(t_ledar, pres_ledar_f(t_ledar), label = 'LeDaR Pressure')
    ax.set_xlim(0,4)
    ax.set_xticks(np.linspace(0,4,9))
    ax.set_ylim(950,1025)
    ax.legend(loc = 'lower right')
    fig.savefig('LeDaR_PIV_pressure_comparison.png', dpi = 400)
#     # #create the folder to save the data
#     # SAVE = '%d_pa' %pressure_case[i] + os.sep
#     # if not os.path.exists(SAVE):
#     #     os.mkdir(SAVE)
    
# #     #save the solutions
# #     np.savetxt(SAVE + 'Sol_Normal_%d.txt' %pressure_case[i], solution_normal)
    
    