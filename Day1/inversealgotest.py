import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

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

#Define ODE
def ode(y, t, k):
    dydt = - k * y +1 
    return dydt

#Set start Value
y0 = 2

#Set t
t = np.linspace(0, data[len(data)-1][0], len(data))

#Initial Guess
a0 = 1

solution = minimize(meanroot, a0)
print(solution)