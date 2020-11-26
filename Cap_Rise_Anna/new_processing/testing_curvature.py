# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:38:55 2020

@author: manuel
@description: Code to check the integration of the curvature for the laplace equation
"""


import numpy as np
import matplotlib.pyplot as plt

# set up a test circle, this one has an angle of 45 Degrees at the edges
x = np.linspace(-1,1, 1001)
R = np.sqrt(2)
y = -np.sqrt(R**2-x**2)

# calculate the surface tension force via the contact angle
surface_force = 2*np.cos(np.pi/4)

# calculate the gradient to get the first derivative
dydx = np.gradient(y,x)
# and again for the second derivative
ddyddx = np.gradient(dydx, x)

# calculate the curvature
curvature = ddyddx/(1+dydx**2)**1.5
# we have to pad near the edges because the gradients get weird over there
curvature[0] = curvature[2]
curvature[1] = curvature[2]
curvature[-1] = curvature[-3]
curvature[-2] = curvature[-3]

# calculate the surface tension force via the integrated curvature
surface_force2 = np.trapz(curvature, x)

# print to compare
print(surface_force2)
print(surface_force)