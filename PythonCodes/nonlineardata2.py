import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def ode(ye, t):
    y = ye[0]
    z = ye[1]
    dydt = z
    dzdt = -0.3 * y * t - 0.5 * z - 2
    return [dydt, dzdt]

y0 = np.array([1 , 0.5])

tsteps = 100

t = np.linspace(0, 20, tsteps+1)
y = odeint(ode, y0, t)
arr = np.zeros((tsteps+1, 2), dtype = np.float)

for i in range(0, tsteps+1):
    arr[i][0] = i/tsteps
    arr[i][1] = y[i][0]

np.savetxt('exportdatanonlin2.csv', arr, delimiter=',')

data = np.loadtxt('exportdatanonlin2.csv', delimiter=',')
plt.plot(t, y[:,0])
plt.xlabel('time')
plt.ylabel('y(t)')