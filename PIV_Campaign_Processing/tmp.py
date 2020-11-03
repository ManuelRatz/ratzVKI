# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:54:10 2020

@author: manue
"""


import numpy as np

a = np.genfromtxt('wallcuts.txt', dtype = str)
names = a[:,0].astype(np.str)
wallcuts = a[:,1:].astype(np.float)

first_name = names[0]
second_name = names[1]

Name = 'F_h1_f1000_1_q'
for j in range(0, len(names)):
    if(Name == names[j]):
        wallcut_left, wallcut_right = wallcuts[j,:]