#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:02:56 2020

@author: ratz
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# load test data
Fol_In = 'test_images' + os.sep + 'raw_images' + os.sep + 'Txts_advanced_fitting'\
    + os.sep
Fol_Old = '..' + os.sep + 'experimental_data' + os.sep + '1500_pascal' + os.sep


h_new = np.genfromtxt(Fol_In + 'Displacement.txt')
h_old = np.genfromtxt(Fol_Old + 'avg_height_1500.txt')
h_r = np.genfromtxt(Fol_In + 'Displacement_CLsx.txt')
h_r_old = np.genfromtxt(Fol_Old + 'cl_r_1500.txt')
h_l = np.genfromtxt(Fol_In + 'Displacement_CLdx.txt')
rca = np.genfromtxt(Fol_In + 'RCA.txt')
lca = np.genfromtxt(Fol_In + 'LCA.txt')
rca_old = np.genfromtxt(Fol_Old + 'rca_1500.txt')
idx = np.argmax(rca > 0)
rca = rca[idx:]
lca = lca[idx:]
h_new = h_new[idx:]
h_r = h_r[idx:]
h_l = h_l[idx:]



# # rca_smooth = savgol_filter(rca_deg, 15, 2, axis = 0)

# # plt.plot(h_old)
# # plt.plot(h_new*0.001+0.074)
# # plt.plot(h_l)
# # plt.plot(h_r)
# # plt.plot(rca_deg, label = 'new')
# # plt.plot(rca_old*180/np.pi, label = 'old')
# # plt.plot(rca_smooth, label = 'smoothed')
# plt.plot(h_r, label = 'right')
# plt.plot(h_l, label = 'left')
# plt.plot(h_new, label = 'average')
# # plt.plot(h_r_old*1000-74, label = 'old')
# plt.legend()