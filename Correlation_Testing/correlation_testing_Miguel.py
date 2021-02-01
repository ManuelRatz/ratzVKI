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
Case = 'H_P2000_A'
vel, vel_cl, acc, acc_cl, ca = load_file(Fol_In, Case)

plt.figure()
plt.scatter(vel*mu_h/sigma_h, ca, c = acc, marker = 'o', s = 0.25)
plt.title('Capillary vs Theta')
plt.colorbar()

# plt.figure()
# plt.scatter(acc, ca, c = vel, marker = 'o', s = 0.25)
# plt.colorbar()
# plt.title('Acceleration vs Theta')