import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def ode(ye, t):
    y = ye
    dydt = - 3 * pow(y, 5)
    return dydt

y0 = 1

tsteps = 100

t = np.linspace(0, 20, tsteps+1)
y = odeint(ode, y0, t)
arr = np.zeros((tsteps+1, 2), dtype = np.float)

for i in range(0, tsteps+1):
    arr[i][0] = i/tsteps
    arr[i][1] = y[i][0]

np.savetxt('exportdatanonlin.csv', arr, delimiter=',')

data = np.loadtxt('exportdatanonlin.csv', delimiter=',')
plt.plot(t, y[:,0])
plt.show()
plt.xlabel('time')
plt.ylabel('y(t)')