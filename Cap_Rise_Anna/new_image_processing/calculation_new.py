"""
@author ratz
@description: calculate the height taken from the contact angle calculated
with the new settings for the image processing and do some simple plots
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
    FOLDER = 'data_1500_pa' + os.sep 
    ca_load = np.genfromtxt(FOLDER + 'rca_%d.txt' %pressure_case[i])
    pressure_load = np.genfromtxt(FOLDER + 'pressure_%d.txt' %pressure_case[i])
    h_avg_load = np.genfromtxt(FOLDER + 'avg_height_%d.txt' %pressure_case[i])
    
    # smooth the data 
    h_smoothed = savgol_filter(h_avg_load, 9, 2, axis = 0)
    pressure_smoothed = savgol_filter(pressure_load, 55, 3, axis = 0)
    ca_load = savgol_filter(ca_load, 15, 3, axis = 0)
    
    #set up the timesteps
    t = np.arange(0, len(ca_load)/500, 0.002)
    
    #set up the interpolation functions
    pres_inter_load = interpolate.splrep(t, pressure_load)
    def pres(t):
        return interpolate.splev(t, pres_inter_load)
    ca_inter = interpolate.splrep(t, ca_load)
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

    # calculate the solution
    solution_normal = odeint(ode_normal, X0, t)
    
    # plot the height
    fig, ax = plt.subplots(figsize = (8,5))
    ax.set_xlim(0,4)
    ax.set_ylim(h_smoothed[0]*1000, ((np.max(solution_normal[:,1])*1000)+2))
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    plt.plot(t, solution_normal[:,1]*1000)
    plt.plot(t, h_smoothed*1000)
    fig.savefig(SAVE_IMAGES + 'transient_height_comparison_1500.png', dpi=100)
    
    # calculate the accelerations
    acc_normal =  np.zeros((solution_normal.shape[0], 6))
    U = solution_normal[:, 0]
    Y = solution_normal[:, 1]
    acc_normal[:, 0] = -g
    acc_normal[:, 1] = -12 * (l + delta)*mu*U*Y/(rhoL*l*delta**2) * Y**(-1)
    acc_normal[:, 2] = (pres(t)/rhoL)* Y**(-1)
    acc_normal[:, 3] = (2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta))* Y**(-1)
    acc_normal[:, 4] = - U* abs(U) * Y**(-1)
    acc_normal[:, 5] = acc_normal[:,0]+acc_normal[:,1]+acc_normal[:,2]+acc_normal[:,3]+acc_normal[:,4]
    
    fig, ax = plt.subplots(figsize = (8,5))
    ax.set_xlim(0,4)
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    plt.plot(t, acc_normal[:,0]+acc_normal[:,2])
    plt.plot(t, acc_normal[:,3])
    plt.plot(t, acc_normal[:,1])
    plt.plot(t, acc_normal[:,4])
    