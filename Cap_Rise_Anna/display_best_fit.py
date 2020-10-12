#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:52:14 2020

@author: ratz
@desciption: this code loads the RMS array and plots its minimum value compared
to the height of the data to get the currently best approximation of the system
"""
import numpy as np                  # for array operations
import matplotlib.pyplot as plt     # for plotting the result
from scipy.integrate import odeint  # for solving the ode
import os                           # for setting up data paths
from scipy import interpolate       # for setting the interpolation functions

# load the raw data
rms = np.genfromtxt('first_iteration.txt')
Fol_In = 'experimental_data' + os.sep + '1500_pascal' + os.sep
pressure = np.genfromtxt(Fol_In + 'pressure_1500.txt')
height = np.genfromtxt(Fol_In + 'avg_height_1500.txt')

# set the parameters of the ODE and the timesteps
zeta = np.linspace(0.1, 0.3, 21)
omega0 = np.linspace(5, 20, 31)
rho = 1000 # density of water
g = 9.81  # gravitational acceleration
t = np.linspace(0, 4, 2001, endpoint=True)

# set the initial conditions
h0 = height[0] # initial height
vel = np.gradient(height)/0.002 # calculate velocity array
dh0 = vel[0] # initial velocity
X0 = np.array([h0, dh0])

# define the pressure function
pres_inter = interpolate.splrep(t, pressure)
def pres(t):
    return interpolate.splev(t, pres_inter)

# define the ode
def ode(X0, t, idx_w, idx_z):
    y = X0[0] # give initial height
    z = X0[1] # give initial velocity
    dydt = z
    dzdt = -2*omega0[idx_w]*zeta[idx_z]*z - omega0[idx_w]**2*y + omega0[idx_w]**2*pres(t)/(rho*g)
    return [dydt, dzdt]

# define the Cost Function
def RMS(solution, data):
    return np.sqrt(np.sum(1/len(data)*(solution-data)**2))

best_z, best_w = np.where(rms == np.min(rms))
best_sol = odeint(ode, X0, t, args=(best_w, best_z))
fig, ax = plt.subplots()
ax.set_xlim(0, 4)
ax.set_ylim(height[0]*1000, np.amax(height)*1000+2)
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$h$[mm]')
ax.plot(t, height*1000)
ax.plot(t, best_sol[:,0]*1000)
plt.savefig('current_best_approx.png', dpi = 100)
    