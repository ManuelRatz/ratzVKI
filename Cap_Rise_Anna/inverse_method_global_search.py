#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:26:37 2020

@author: ratz
@description: this code tries to find a second order system to fit the data of 
the capillary rise experiment. This is done by creating a grid of 20*30 values 
for zeta and omega0 respectively, solving the ODE for each step and calculating
the RMS of the data and the ODE solution. The initial conditions are the same for
each case.

Afterwards we create a 3d plot of the RMS to get an idea where the minimum might be
"""
import numpy as np                  # for array operations
import matplotlib.pyplot as plt     # for plotting the result
from scipy.integrate import odeint  # for solving the ode
import os                           # for setting up data paths
from scipy import interpolate       # for setting the interpolation functions

# load the raw data
Fol_In = 'experimental_data' + os.sep + '1500_pascal' + os.sep
pressure = np.genfromtxt(Fol_In + 'pressure_1500.txt')
height = np.genfromtxt(Fol_In + 'avg_height_1500.txt')

# set the parameters of the ODE and the timesteps
"""
Note: Zeta has to be between 0 and 1, playing with the variables showed that
only values between 0.1 and 0.3 lead to a similar course of the graph. Omega0
can be estimated by the data to be ~10 but to be sure we look at values between
5 and 20
"""
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
"""
Note: For now we only take the pressure as the driving force on the right side
of the equation and ignore the contact angle
"""
def ode(X0, t, idx_w, idx_z):
    y = X0[0] # give initial height
    z = X0[1] # give initial velocity
    dydt = z
    dzdt = -2*omega0[idx_w]*zeta[idx_z]*z - omega0[idx_w]**2*y + omega0[idx_w]**2*pres(t)/(rho*g)
    return [dydt, dzdt]

# define the Cost Function
def RMS(solution, data):
    return np.sqrt(np.sum(1/len(data)*(solution-data)**2))

# calculate the RMS for every configuration of zeta and omega0
rms = np.zeros((len(zeta), len(omega0))) # set up the array
# iterate over the array
for idx_z in range(0, len(zeta)):
    for idx_w in range(0, len(omega0)):
        sol = odeint(ode, X0, t, args=(idx_w, idx_z)) # calculate the solution
        rms[idx_z][idx_w] = RMS(sol[:,0], height) # save the rms in the array
    # update the user
    print('Finished %d of %d columns' %((idx_z+1), len(zeta))) 

# plot the result
W, Z = np.meshgrid(omega0, zeta) # make a 2d grid consisting of omega0 and zeta
fig = plt.figure() # create the figure
ax = plt.axes(projection='3d') # define a 3d plot
ax.plot_surface(W, Z, rms, cmap='viridis', edgecolor='none') # make a surface plot

# save the rms grid
np.savetxt('first_iteration.txt', rms)

"""
The resulting plot shows that the minimum for each value of zeta is in the area
of an omega0 between 8.5 and 10.5. This is further looked at in "inverse_method_
local_search.py"
"""





















