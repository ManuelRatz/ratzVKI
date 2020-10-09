"""
@author ratz
@description calculate the height predicted by the model from the experimental
pressure and contact angle signal
"""

import numpy as np
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import odeint


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
    FOLDER = '..' + os.sep + 'experimental_data' + os.sep + '%d_pascal' %pressure_case[i] + os.sep
    lca_load = np.genfromtxt(FOLDER + 'lca_%d.txt' %pressure_case[i])
    rca_load = np.genfromtxt(FOLDER + 'rca_%d.txt' %pressure_case[i])
    h_cl_l_load = np.genfromtxt(FOLDER + 'cl_l_%d.txt' %pressure_case[i])
    h_cl_r_load = np.genfromtxt(FOLDER + 'cl_r_%d.txt' %pressure_case[i])
    pressure_load = np.genfromtxt(FOLDER + 'pressure_%d.txt' %pressure_case[i])
    h_avg_load = np.genfromtxt(FOLDER + 'avg_height_%d.txt' %pressure_case[i])
    
    # calculate the averages
    ca = rca_load
    h_cl = (h_cl_l_load + h_cl_r_load) / 2

    # smooth the data 
    ca_smoothed =  ca# savgol_filter(ca, 9, 3, axis = 0)
    h_cl_smoothed = savgol_filter(h_cl, 105, 3, axis = 0)
    h_smoothed = savgol_filter(h_avg_load, 9, 2, axis = 0)
    pressure_smoothed = savgol_filter(pressure_load, 55, 3, axis = 0)
    
    #set up the timesteps
    t = np.arange(0, len(ca)/500, 0.002)
    
    # here we set up the 4 different pressure signals
    step_pressure = np.zeros(len(pressure_smoothed)) + pressure_step[i]
    advanced_step_pressure = np.copy(pressure_smoothed)
    idx = np.argmax(pressure_smoothed > pressure_step[i])
    advanced_step_pressure[idx:] = pressure_step[i]
    
    #set up the interpolation functions

    pres_inter_load = interpolate.splrep(t, pressure_load)
    def pres(t):
        return interpolate.splev(t, pres_inter_load)
    ca_inter = interpolate.splrep(t, ca_smoothed)
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
    def ode_normal(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres(t)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_novisc(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y + pres(t)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_noca(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres(t)/rhoL\
                          - U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_nousquare(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres(t)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta))
        dydt = U
        return [dudt, dydt]
    
    #calculate the solutions
    sol_normal = odeint(ode_normal, X0, t)
    sol_novisc = odeint(ode_novisc, X0, t)
    sol_noca = odeint(ode_noca, X0, t, )
    sol_nousquare = odeint(ode_nousquare, X0, t)
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, sol_normal[:, 1]*1000, label = 'Normal')
    plt.plot(t, sol_novisc[:,1]*1000, label = 'No viscosity')
    plt.plot(t, sol_noca[:,1]*1000, label = 'No $\sigma$')
    plt.plot(t, sol_nousquare[:,1]*1000, label = 'No $u^2$')
    plt.plot(t, h_smoothed*1000, label = 'Data')
    ax.set_xlim([0, 4])
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    plt.legend(ncol = 2)
    plt.savefig('Testing.png')
