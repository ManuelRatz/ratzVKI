import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

## Hello Manuel, Miguel was here. I do not like this file, it can't work.

#Load Data
data = np.loadtxt('exportdata.csv', delimiter=',')

#Calculate Mean Square Root
def meanroot(a):
    y = odeint(ode, y0, t, args=(a,))
    tempsum = 0
    for i in range(len(data)):
        tempsum = tempsum + (data[i][1]-y[i])**2
        tempsum = np.sqrt(tempsum)
    return (np.sqrt(tempsum))

def ode(y, t, k):
    dydt = z
    dzdt = - z / 8 - y
    return