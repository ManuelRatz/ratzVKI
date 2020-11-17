"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Code to compare different refinements for the processing of a rise
"""

import matplotlib.pyplot as plt         # for plotting
import post_processing_functions as ppf # for reshaping the arrays

# set the default plot parameters
ppf.set_plot_parameters()

# set input and output folders
Fol_In_2 = 'C:\\Users\manue\Desktop\Results_16_64_R_h1_f1200_1_p15'
Fol_In_1 = 'C:\\Users\manue\Desktop\Results_12_48_R_h1_f1200_1_p15'


nx1 = ppf.get_column_amount(Fol_In_1)
nx2 = ppf.get_column_amount(Fol_In_2)

IDX1 = 20
IDX2 = 9
n_t = 1
frame0 = 3
for k in range(frame0, frame0+n_t):
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    x1, y1, u1, v1 = ppf.load_txt(Fol_In_1, 3*k+frame0, nx1)
    x2, y2, u2, v2 = ppf.load_txt(Fol_In_2, 3*k+frame0, nx2)
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x1[IDX1,:],v1[IDX1,:], label = 'Fine Grid')
    ax.scatter(x1[IDX1,:],v1[IDX1,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x2[IDX2,:],v2[IDX2,:], label = 'Rough Grid', c='r')
    ax.scatter(x2[IDX2,:],v2[IDX2,:], marker='x', s=(300./fig.dpi)**2, c='r')  
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 3)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(-5,17.5)
    ax.set_xlim(0,270)
    plt.title('Frame %03d' %(3*k))