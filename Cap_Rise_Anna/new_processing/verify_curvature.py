# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:52:28 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import matplotlib.pyplot as plt
import numpy as np

Radius = np.sqrt(2)
Xrange = 1
x = np.linspace(-Xrange, Xrange, 10001)+0.3
y = Radius-np.sqrt(Radius**2-x**2)

deriv = np.gradient(y, x)
angle = np.arctan(deriv[-1])

def integrate_curvature(x, y):
    dydx = np.gradient(y, x)
    ddyddx = np.gradient(dydx, x)
    curvature = ddyddx / (1 + (dydx)**2)**1.5
    ret = np.trapz(curvature, x)/(np.max(x)-np.min(x))
    return ret

curv = integrate_curvature(x, y)
pres_drop = np.cos(angle)
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.set_aspect('equal')