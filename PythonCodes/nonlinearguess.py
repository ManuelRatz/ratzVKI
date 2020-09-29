import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

#Load Data
data = np.loadtxt('exportdatanonlin.csv', delimiter=',')

#Calculate Mean Square Root
def meanroot(a):
    y = odeint(ode, y0, t, args=(a,))
    tempsum = 0
    for i in range(len(data)):
        tempsum = tempsum + (data[i][1]-y[i])**2
    return (np.sqrt(tempsum))

#Define ODE
def ode(ye, t, a):
    y = ye
    dydt = - a[0] * pow(y, a[1])
    return dydt

#Start Values
y0 = 1

#Initial Guess
a0 = np.array([1,1])

#create t points
t = np.linspace(0, 20, len(data))

solution = minimize(meanroot, a0)
print(solution)