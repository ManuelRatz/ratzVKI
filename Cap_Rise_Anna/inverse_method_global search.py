"""
@author: ratz
@description:
This code tries the inverse method for a second order linear system with the
data of the capillary rise measured by Anna. As the optimizer might get stuck
in a local optimum, we generate a grid that gives values for zeta and omega0
and plot the cost function of the Root Mean Square to get an idea for the 
amount of solutions and the map of the cost function
"""

import numpy as np                   # for operations with arrays
import matplotlib.pyplot as plt      # for plotting things
from scipy.optimize import minimize  # for using the optimizer later (potentially)
import os                            # for creating file paths
from scipy import interpolate        # for interpolating the pressure data
from scipy.integrate import odeint   # for solving an ODE

# Set the constants
rho = 1000
g = 9.81

# Load the data
Fol_In = 'experimental_data' + os.sep + '1500_pascal' + os.sep
pressure = np.genfromtxt(Fol_In + 'pressure_1500.txt')
height = np.genfromtxt(Fol_In + 'avg_height_1500.txt')
idx = np.argmax(height > 0)
pressure = pressure[idx:]
height = height[idx:]

# Define the Cost Function
def RMS(solution, data):
    return np.sqrt(np.sum(1/len(data)*(solution-data)**2))

# Define the ODE
def ODE(ye, t, vari):
    y = ye[0]
    z = ye[1]
    dydt = z
    dzdt = - 2*vari[0]*vari[1]*z - y*(vari[1])**2 + pres(t)*(vari[1])**2/(rho*g)
    return [dydt, dzdt]

# Set up the timesteps
fps = 500
t = np.arange(0, len(pressure)/fps, 1/fps)

# Define the pressure function
pres_inter = interpolate.splrep(t, pressure)
def pres(t):
    return interpolate.splev(t, pres_inter)

# Set the initial conditions
h0 = height[0]
dh0 = (height[1]-height[0])/0.002
X0 = np.array([h0, dh0])

# set up the zeta values
zeta = np.arange(0.05, 1, 0.025)

# Set up the omega0 values
omega0 = np.arange(0.1, 20, 0.1)

# Initiate the value matrix
"""
The values are stored in a 3d array, one dimension containing all the zeta 
values and the other the ones for omega0. Odeint will call the function for
every pair of values to get len(zeta)*len(omega0) results
"""
matr = np.zeros((2, len(omega0), len(zeta)))
matr[0, :, :] = np.tile(zeta, (len(omega0),1))
matr = matr.transpose(0,2,1)
matr[1, :, :] = np.tile(omega0, (len(zeta),1))
matr = matr.transpose(0,2,1)

# Set up the RMS array
# rms = np.zeros((len(omega0), len(zeta)))

"""
Comment this block to do the calculations, this is very time consuming
"""
# # Call and save the cost function for each value of the grid
# for i in range(0, len(zeta)):
#     for j in range(0, len(omega0)):
#         init = np.array(matr[:, j, i])
#         solution = odeint(ODE, X0, t, args=(init,))
#         rms[j, i] = RMS(solution[:, 0], height)
#     #update user
#     print('Finished %.3f%%' %((i+1)/len(zeta)*100))
rms = np.genfromtxt('data.txt')
result = np.where(rms == np.amin(rms))
sol = odeint(ODE, X0, t, args = (matr[:,5, 96],))