# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:00:14 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\new_processing')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from fractions import Fraction

# define the ode
def ode(values, t, a, b, c, d):
    # a = make_odd(a)
    # b = make_odd(b)
    # print(a)
    y = values[0]
    z = values[1]
    dydt = z
    dzdt = -z * a/b - y*c/d
    return [dydt, dzdt]

# def make_odd(number):
#     number = number//1
#     if number%2 == 0:
#         return number+1
#     return number

def cost(a):
    # print(a)
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    penalty0 = 5*np.abs(a0-a0//1)
    penalty1 = 0 # np.abs(a1-a1//1)
    penalty2 = 5*np.abs(a2-a2//1)
    penalty3 = 0 # np.abs(a3-a3//1)
    # penalty = 0
    # if a1 % 2 != 1:
    #     penalty = 50
    y_loc = odeint(ode, y0, t, args = (a0, a1, a2, a3))[:,1]
    return np.linalg.norm(y_loc - data) + penalty0 + penalty1 + penalty2 + penalty3

# create the time axis
t = np.linspace(0, 20, 1001)
# initial condition
y0 = np.array([0, 1])
# synthetic data
data = odeint(ode, y0, t, args = (329, 523, 7, 9, ))[:,1]

# initial guess
a0 = np.array([2, 6, 5, 8])

def constraint_num1(a):
    return a[0]%2-1
def constraint_num2(a):
    return a[2]%2-1
cons1 = {'type' : 'eq', 'fun' : constraint_num1}
cons2 = {'type' : 'eq', 'fun' : constraint_num2}

constraints = [cons1, cons2]

# calculate the solution with the inverse method
solution = minimize(cost, a0, method = 'Nelder-Mead')
                    # options = {'maxiter' : 100, 'ftol' : 1e-6})
test = Fraction(solution.x[0]/solution.x[1]).limit_denominator()
print(solution.message)
print(solution.fun)
print(solution.x)
print(solution.x[0]/solution.x[1])
print(solution.x[2]/solution.x[3])

# def func(x):
#     return x**2

# def constraint(x):
#     return x%2

# cons = {'type' : 'eq', 'fun' : constraint}

# x = np.linspace(-5, 5, 101)

# plt.figure()
# plt.plot(x, func(x))
# plt.plot(x, constraint(x))

# x0 = 2
# sol = minimize(func, x0 = x0, method = 'SLSQP', constraints = cons)