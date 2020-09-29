import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import imageio

#Define ODE
def osci(ye, t):
    y = ye[0]
    z = ye[1]
    dydt = z
    dzdt = - z / 4 - y
    return [dydt, dzdt]

#Set start values
y0 = np.array([1, 0])

#pick number of time steps
tsteps = 100

#create data array
t = np.linspace(0, 20, tsteps+1)
y = odeint(osci, y0, t)
arr = np.zeros((tsteps+1, 2), dtype = np.float)
for i in range(0, tsteps+1):
    arr[i][0] = i/tsteps
    arr[i][1] = y[i][0]

#Export array
np.savetxt('exportdatasecord.csv', arr, delimiter=',')

#print data to console
plt.plot(t, y[:,0])
plt.xlabel('time')
plt.ylabel('y(t)')
plt.savefig('correct_ode_solution.png', dpi = 100)