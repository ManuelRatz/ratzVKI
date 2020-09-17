import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def osci(ye, t):
    y = ye[0]
    z = ye[1]
    dydt = z
    dzdt = - z / 4 - y
    return [dydt, dzdt]

y0 = [1, 0]

tsteps = 500

t = np.linspace(0, 20, tsteps+1)
y = odeint(osci, y0, t)
arr = np.zeros((tsteps+1, 2), dtype = np.float)

for i in range(0, tsteps+1):
    arr[i][0] = i/tsteps
    arr[i][1] = y[i][0]

np.savetxt('exportdata.csv', arr, delimiter=',')

data = np.loadtxt('exportdata.csv', delimiter=',')
plt.plot(t, y[:,0])
plt.xlabel('time')
plt.ylabel('y(t)')