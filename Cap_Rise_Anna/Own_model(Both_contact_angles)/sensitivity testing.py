"""
This file investigates the sensitivity of the solution depending on the initial
conditions. This includes different smoothing options for the velocity and
offsets for the height
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

# Define the ODE function
def ode_normal(X, t):
    U, Y = X
    dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres(t)/rhoL\
                      + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
    dydt = U
    return [dudt, dydt]
    
# Test cases used for evaluation
pressure_case = np.array([1000, 1250, 1500])
# here the pressure steps as measured by the sensor are given
pressure_step = np.array([662.6799, 823.1328, 990.89])

# set up the for loop
for i in range(0, 3):
    # Set up the saving folder for the data
    FOL_OUT = '%d_pa_images' %pressure_case[i] + os.sep
    # Update the user about the status
    print('Calculating pressure %d of %d' %((i+1), len(pressure_case)))
    # load the data
    FOLDER = '..' + os.sep + 'experimental_data' + os.sep + '%d_pascal' %pressure_case[i] + os.sep
    lca_load = np.genfromtxt(FOLDER + 'lca_%d.txt' %pressure_case[i])
    rca_load = np.genfromtxt(FOLDER + 'rca_%d.txt' %pressure_case[i])
    pressure_load = np.genfromtxt(FOLDER + 'pressure_%d.txt' %pressure_case[i])
    h_avg_load = np.genfromtxt(FOLDER + 'avg_height_%d.txt' %pressure_case[i])
    
    # calculate the averages
    ca = (lca_load + rca_load) / 2
    # h_cl = (h_cl_l_load + h_cl_r_load) / 2
    
    # shift the data
    shift_idx = np.argmax(ca > 0)
    ca = ca[shift_idx:]
    h_avg_load = h_avg_load[shift_idx:]
    pressure_load= pressure_load[shift_idx:len(ca)+shift_idx]
    
    # smooth the data 
    ca_smoothed = savgol_filter(ca, 35, 3, axis = 0)
    h_smoothed = savgol_filter(h_avg_load, 9, 3, axis = 0)
    pressure_smoothed = savgol_filter(pressure_load, 151, 3, axis = 0)
    
    # set up the timesteps
    t = np.arange(0, len(ca)/500, 0.002)
    
    step_pressure = np.zeros(len(pressure_smoothed)) + pressure_step[i]
    advanced_step_pressure = np.copy(pressure_smoothed)
    idx = np.argmax(pressure_smoothed > pressure_step[i])
    advanced_step_pressure[idx:] = pressure_step[i]
    
    # set up the interpolation functions
    pres_inter_load = interpolate.splrep(t, pressure_load)
    def pres(t):
        return interpolate.splev(t, pres_inter_load)
    ca_inter = interpolate.splrep(t, ca_smoothed)
    def ca(t):
        return interpolate.splev(t, ca_inter)
    
    """
    First we look at the different heights for offsets of 3 mm for the initial
    height. This has a very big impact on the course of the graph but not on
    the equililbrium height that is calculated by integrating over the last
    second. Thus we take a different approch: First calculate the solution for
    the given initial height, no offset. Afterwards calculate the equilibrium
    height of the data and the prediction, calculate the offset. Finally
    calculate the new solution for this new value of the height considering 
    the offset.
    """
    # set the inital height (with offset)
    h0 = h_smoothed[0]
    h1 = h0 + 0.003
    h2 = h0 - 0.003
    # calculate the initial velocity
    velocity = np.gradient(h_smoothed)
    dh0 = velocity[0]/(0.002)
    X0 = np.array([dh0, h0])
    X1 = np.array([dh0, h1])
    X2 = np.array([dh0, h2])
    
    # calculate the solutions for the initial conditions with different heights
    solution0 = odeint(ode_normal, X0, t)
    solution1 = odeint(ode_normal, X1, t)
    solution2 = odeint(ode_normal, X2, t)
    """
    # create the plot comparing the height
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    ax.set_xlim([0, 4])
    ax.set_ylim([h2*1000, np.amax(solution2[:,1]*1000)+2])
    plt.plot(t, solution0[:,1]*1000, c = 'k', label = 'No deviation')
    plt.plot(t, solution1[:,1]*1000, c = 'b', label = '3mm above')
    plt.plot(t, solution2[:,1]*1000, c = 'c', label = '3mm below')
    plt.plot(t, h_smoothed*1000+3, c = 'r', label = 'Experimental data')
    plt.legend(loc = 'lower right')
    plt.title('Comparison of different offsets for the height (%d Pa)' %pressure_case[i])
    plt.savefig(FOL_OUT + 'Sensitivity_init_height_%d.png' %pressure_case[i])
    
    # Calculate the equilibrium height by integrating over the last second for
    # the solution that does not have an offset
    h_eq_0 = np.sum(solution0[1500:, 1])/500
    h_eq_data = np.sum(h_avg_load[1500:])/500
    offset_height = h_eq_0 - h_eq_data
    
    # set the new initial value for the height considering the offset
    h_off = h_smoothed[0] + offset_height
    X_off = np.array([dh0, h_off])
    sol_offset = odeint(ode_normal, X_off, t)
    
    fig, ax = plt.subplots(figsize = (8,5))
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    ax.set_xlim([0, 4])
    ax.set_ylim([h_off*1000, np.amax(sol_offset[:,1]*1000)+2])
    plt.plot(t, sol_offset[:, 1]*1000, label = 'Model prediction')
    plt.plot(t, (h_smoothed+offset_height)*1000, label = 'Data with offset')
    plt.legend(loc = 'lower right')
    plt.title('Height comparison for calculated offset (%d Pa)' %pressure_case[i])
    plt.savefig(FOL_OUT + 'Height_comparison_offset_%d.png' %pressure_case[i])
    """
    """
    Here we are looking at different velocities for the initial conditions. This
    is done by applying different smoothing options for the height and then
    calculating the gradient as done above
    """
    # use two smoothing options, light and heavy
    h_avg_light = savgol_filter(h_avg_load, 7, 3, axis = 0)
    h_avg_heavy = savgol_filter(h_avg_load, 125, 1, axis = 0)
    vel_data = np.gradient(h_avg_load)/0.002
    vel_light = np.gradient(h_avg_light)/0.002
    vel_heavy = np.gradient(h_avg_heavy)/0.002
    
    # set up the initial conditions
    X_0 = np.array([vel_data[0], h_avg_load[0]])
    X_1 = np.array([vel_light[0], h_avg_load[0]])
    X_2 = np.array([vel_heavy[0], h_avg_load[0]])
    
    # calculate the solutions
    solution_vel_0 = odeint(ode_normal, X_0, t)
    solution_vel_1 = odeint(ode_normal, X_1, t)
    solution_vel_2 = odeint(ode_normal, X_2, t)
    
    # set the initial conditions
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    ax.set_xlim([0, 4])
    # ax.set_ylim([71, 118])
    plt.plot(t, solution_vel_0[:,1]*1000, c = 'k', label = 'No smoothing')
    plt.plot(t, solution_vel_1[:,1]*1000, c = 'b', label = 'Light smoothing')
    plt.plot(t, solution_vel_2[:,1]*1000, c = 'c', label = 'Heavy smoothing')
    plt.plot(t, h_smoothed*1000, c = 'r', label = 'Experimental data')
    plt.legend(loc = 'lower right')
    plt.title('Comparison of different smoothing options for the velocity (%d Pa)' %pressure_case[i])
    plt.savefig(FOL_OUT + 'Sensitivity_init_vel_smoothing_%d.png' %pressure_case[i])
    