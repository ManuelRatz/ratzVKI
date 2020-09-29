import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

#Load Data
data = np.loadtxt('exportdatanonlin2.csv', delimiter=',')

#Calculate Mean Square Root
def meanroot(a):
    y = odeint(ode, y0, t, args=(a,))
    tempsum = 0
    for i in range(len(data)):
        tempsum = tempsum + (data[i][1]-y[i][0])**2
    return (np.sqrt(tempsum))

#Define ODE
def ode(ye, t, a):
    y = ye[0]
    z = ye[1]
    dydt = z
    dzdt = - a[0] * y * t - a[1] * z - 2
    return [dydt, dzdt]

#Initial Value
y0 = np.array([1, 0.5])

#Initial Guess
a0 = np.array([1,1])

#create t points
t = np.linspace(0, 20, len(data))

y = odeint(ode, y0, t, args=(a0,))


solution = minimize(meanroot, a0)
print(solution)