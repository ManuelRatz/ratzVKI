# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:23:38 2021

@author: Manuel Ratz
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

""" Liquid properties """
# water:
mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724

# hfe
mu_h = 0.000667
rho_h = 1429.41965
sigma_h = 0.01391

t = np.linspace(0, 4, 2000, endpoint = False)

Fol_In = 'Data'

def load_file(folder, case):
    file = os.path.join(folder, case + '.txt')
    data = np.genfromtxt(file)
    vel = data[:,0]
    vel_cl = data[:,1]
    acc = data[:,2]
    acc_cl = data[:,3]
    ca = data[:,4]
    return vel, vel_cl, acc, acc_cl, ca

#%%
Case = 'W_P1500_C'
vel, vel_cl, acc, acc_cl, ca = load_file(Fol_In, Case)

# plt.figure()
plt.scatter(vel_cl*mu_h/sigma_h, ca, c = acc, marker = 'o', s = 0.25)
# plt.title('Capillary vs Theta')
# plt.colorbar()

# plt.figure()
# plt.scatter(acc, ca, c = vel, marker = 'o', s = 0.25)
# plt.colorbar()
# plt.title('Acceleration vs Theta')

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
ax.scatter(acc, vel, ca, s=0.2, marker = 'o')
ax.set_xlabel('acceleration')
ax.set_ylabel('velocity')
ax.set_zlabel('contact angle')
# ax.legend()

fig.tight_layout()
