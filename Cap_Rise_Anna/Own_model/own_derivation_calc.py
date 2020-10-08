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
for i in range(0, 3):
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
    
    # shift the data
    shift_idx = np.argmax(ca > 0)
    ca = ca[shift_idx:]
    h_cl = h_cl[shift_idx:]
    h_avg_load = h_avg_load[shift_idx:]
    pressure_load= pressure_load[shift_idx:len(ca)+shift_idx]
    
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
    pres_inter = interpolate.splrep(t, pressure_smoothed)
    def pres_smoothed(t):
        return interpolate.splev(t, pres_inter)
    pres_inter_load = interpolate.splrep(t, pressure_load)
    def pres(t):
        return interpolate.splev(t, pres_inter_load)
    pres_step_inter = interpolate.splrep(t, step_pressure)
    def pres_step(t):
        return interpolate.splev(t, pres_step_inter)
    pres_step_adv_inter = interpolate.splrep(t, advanced_step_pressure)
    def pres_step_adv(t):
        return interpolate.splev(t, pres_step_adv_inter)
    ca_inter = interpolate.splrep(t, ca_smoothed)
    def ca(t):
        return interpolate.splev(t, ca_inter)
    
    #create the folder to save the images
    SAVE_IMAGES = '%d_pa_images' %pressure_case[i] + os.sep
    if not os.path.exists(SAVE_IMAGES):
        os.mkdir(SAVE_IMAGES)
        
    #Plot the pressure courses and save them
    fig, ax = plt.subplots(figsize = (8, 5))
    # plt.plot(t, pres(t), label = 'Unfiltered')
    plt.plot(t, pres_smoothed(t), label = 'Filtered')
    plt.plot(t, pres_step(t), label = 'Step')
    plt.plot(t, pres_step_adv(t), label = 'Advanced step')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$p$[Pa]')
    ax.set_xlim([0, 4])
    plt.legend(loc = 'lower right')
    # plt.title('Comparison of different pressure signals')
    plt.savefig(SAVE_IMAGES + 'Pressure_signals_%d.png' %pressure_case[i])
    
    
    # Do the same plot just without the step and adv step pressures
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, pres(t), label = 'Unfiltered')
    plt.plot(t, pres_smoothed(t), label = 'Filtered')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$p$[Pa]')
    ax.set_xlim([0, 4])
    plt.legend(loc = 'lower right')
    # plt.title('Comparison of different pressure signals')
    plt.savefig(SAVE_IMAGES + 'Pressure_signals_filter_unfilter_%d.png' %pressure_case[i])
    
    
    """
    In the Latex document make a not that the filtered and unfiltered version of
    the pressure are the same because odeint has a filter built in. To save time
    and space the unfiltered version will not be considered from now on
    """
    
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
    def ode_filter(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_smoothed(t)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_step(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_step(t)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_step_adv(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_step_adv(t)/rhoL\
                          + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
        dydt = U
        return [dudt, dydt]
    
    #calculate the solutions
    solution_normal = odeint(ode_normal, X0, t)
    solution_filter = odeint(ode_filter, X0, t)
    solution_step = odeint(ode_step, X0, t)
    solution_step_adv = odeint(ode_step_adv, X0, t)
    
    #create the folder to save the data
    SAVE = '%d_pa' %pressure_case[i] + os.sep
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    
    #save the solutions
    np.savetxt(SAVE + 'Sol_Normal_%d.txt' %pressure_case[i], solution_normal)
    np.savetxt(SAVE + 'Sol_Filter_%d.txt' %pressure_case[i], solution_filter)
    np.savetxt(SAVE + 'Sol_Step_%d.txt' %pressure_case[i], solution_step)
    np.savetxt(SAVE + 'Sol_Step_adv_%d.txt' %pressure_case[i], solution_step_adv)
    
    """
    Here the accelerations are calculated for each of the 3 configurations
    For ease of access and comparison they are stored in a matrix, with each column
    representing one of the acceleration terms. The order is as follows:
        0 - gravity             (this will for the current model always be -g)
        1 - viscous term        (this is still not 100% safe, see the meeting with domenico tomorrow, delete after 23.09.)
        2 - pressure term
        3 - contact angle/surface tension term
        4 - velocity squared term
        5 - total acceleration
    """
    
    #initialize the acceleration matrices
    acc_normal =  np.zeros((solution_normal.shape[0], 6))
    acc_filter = np.zeros((solution_normal.shape[0], 6))
    acc_step = np.zeros((solution_normal.shape[0], 6))
    acc_step_adv =np.zeros((solution_normal.shape[0], 6))
     
    #For the unfiltered pressure
    U = solution_normal[:, 0]
    Y = solution_normal[:, 1]
    acc_normal[:, 0] = -g
    acc_normal[:, 1] = -12 * (l + delta)*mu*U*Y/(rhoL*l*delta**2) * Y**(-1)
    acc_normal[:, 2] = (pres(t)/rhoL)* Y**(-1)
    acc_normal[:, 3] = (2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta))* Y**(-1)
    acc_normal[:, 4] = - U* abs(U) * Y**(-1)
    acc_normal[:, 5] = acc_normal[:,0]+acc_normal[:,1]+acc_normal[:,2]+acc_normal[:,3]+acc_normal[:,4]
    
    #For the filtered pressure
    U = solution_filter[:, 0]
    Y = solution_filter[:, 1]
    acc_filter[:, 0] = -g
    acc_filter[:, 1] = -12 * (l + delta)*mu*U*Y/(rhoL*l*delta**2) * Y**(-1)
    acc_filter[:, 2] = (pres_smoothed(t)/rhoL)* Y**(-1)
    acc_filter[:, 3] = (2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta))* Y**(-1)
    acc_filter[:, 4] = - U* abs(U) * Y**(-1)
    acc_filter[:, 5] = acc_filter[:,0]+acc_filter[:,1]+acc_filter[:,2]+acc_filter[:,3]+acc_filter[:,4]
    
    #For the step pressure
    U = solution_step[:, 0]
    Y = solution_step[:, 1]
    acc_step[:, 0] = -g
    acc_step[:, 1] = -12 * (l + delta)*mu*U*Y/(rhoL*l*delta**2) * Y**(-1)
    acc_step[:, 2] = (pres_step(t)/rhoL)* Y**(-1)
    acc_step[:, 3] = (2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta))* Y**(-1)
    acc_step[:, 4] = - U* abs(U) * Y**(-1)
    acc_step[:, 5] = acc_step[:,0]+acc_step[:,1]+acc_step[:,2]+acc_step[:,3]+acc_step[:,4]
    
    #For the advanced step pressure
    U = solution_step_adv[:, 0]
    Y = solution_step_adv[:, 1]
    acc_step_adv[:, 0] = -g
    acc_step_adv[:, 1] = -12 * (l + delta)*mu*U*Y/(rhoL*l*delta**2) * Y**(-1)
    acc_step_adv[:, 2] = (pres_step_adv(t)/rhoL)* Y**(-1)
    acc_step_adv[:, 3] = (2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta))* Y**(-1)
    acc_step_adv[:, 4] = - U* abs(U) * Y**(-1)
    acc_step_adv[:, 5] = acc_step_adv[:,0]+acc_step_adv[:,1]+acc_step_adv[:,2]+acc_step_adv[:,3]+acc_step_adv[:,4]
    
    #Save the acceleration arrays
    np.savetxt(SAVE + 'Acc_Normal_%d.txt' %pressure_case[i], acc_normal)
    np.savetxt(SAVE + 'Acc_Filter_%d.txt' %pressure_case[i], acc_filter)
    np.savetxt(SAVE + 'Acc_Step_%d.txt' %pressure_case[i], acc_step)
    np.savetxt(SAVE + 'Acc_Step_adv_%d.txt' %pressure_case[i], acc_step_adv)