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
Fol_In_1 = 'C:\\Users\manue\Desktop\\text_files_24_96'
Fol_In_2 = 'C:\\Users\manue\Desktop\\text_files_12_48'
Fol_In_1 = 'C:\\Users\manue\Desktop\\text_files_16_64'
Fol_In_4 = 'C:\\Users\manue\Desktop\\text_files_16_64_Run_2'
Fol_In_2 = 'C:\PIV_Processed\Images_Processed\Results_16_64_R_h1_f1200_1_p15\\text_files'

nx1 = ppf.get_column_amount(Fol_In_1)
nx2 = ppf.get_column_amount(Fol_In_2)
# nx3 = ppf.get_column_amount(Fol_In_3)
# nx4 = ppf.get_column_amount(Fol_In_4)

IDX1 = 4
IDX2 = 4
IDY1 = 20
IDY2 = 20

# IDX1 = 3
# IDX2 = 7
# IDX3 = 5
# IDX4 = 5

# IDY1 = 1
# IDY2 = 3
# IDY3 = 2
# IDY4 = 2

n_t = 1
frame0 = 309

for k in range(0, n_t):
    load_index = 3*k+frame0
    print('Image ' + str(k+1) + ' of ' + str(n_t))
    x1, y1, u1, v1 = ppf.load_txt(Fol_In_1, load_index, nx1)
    x2, y2, u2, v2 = ppf.load_txt(Fol_In_2, load_index, nx2)
    # x3, y3, u3, v3 = ppf.load_txt(Fol_In_3, load_index, nx3)
    # x4, y4, u4, v4 = ppf.load_txt(Fol_In_4, load_index, nx4)
    # create the figure
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x1[IDX1,:],v1[IDX1,:], label = 'linear')
    ax.scatter(x1[IDX1,:],v1[IDX1,:], marker='x', s=(300./fig.dpi)**2)
    ax.plot(x2[IDX2,:],v2[IDX2,:], label = 'circular', c='r')
    ax.scatter(x2[IDX2,:],v2[IDX2,:], marker='x', s=(300./fig.dpi)**2, c='r')
    # ax.plot(x3[IDX3,:],v3[IDX3,:], label = '16 64 Rough Start', c='y')
    # ax.scatter(x3[IDX3,:],v3[IDX3,:], marker='x', s=(300./fig.dpi)**2, c='r')
    # ax.plot(x4[IDX4,:],v4[IDX4,:], label = '16 64 Fine Start', c='g')
    # ax.scatter(x4[IDX4,:],v4[IDX4,:], marker='x', s=(300./fig.dpi)**2, c='r')
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 2)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(0,15)
    ax.set_xlim(0,270)
    plt.title('Frame %03d' %(load_index))
    fig.savefig('circ_vs_lin_horizontal.jpg', dpi = 400)
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(y1[:,IDY1],v1[:,IDY1], label = 'linear')
    ax.scatter(y1[:,IDY1],v1[:,IDY1], marker='x', s=(300./fig.dpi)**2)
    ax.plot(y2[:,IDY2],v2[:,IDY2], label = 'circular', c='r')
    ax.scatter(y2[:,IDY2],v2[:,IDY2], marker='x', s=(300./fig.dpi)**2, c='r')
    # ax.plot(y3[:,IDY3],v3[:,IDY3], label = '16 64 Rough Start', c='y')
    # ax.scatter(y3[:,IDY3],v3[:,IDY3], marker='x', s=(300./fig.dpi)**2, c='y')
    # ax.plot(y4[:,IDY4],v4[:,IDY4], label = '16 64 Fine Start', c='g')
    # ax.scatter(y4[:,IDY4],v4[:,IDY4], marker='x', s=(300./fig.dpi)**2, c='g')
    ax.grid(b = True, lw = 1)
    ax.legend(loc = 'lower center', ncol = 2)
    ax.set_ylabel('v[px/frame]')
    ax.set_xlabel('$x$[px]')
    ax.set_ylim(14,16)
    ax.set_xlim(0,1230)
    plt.title('Frame %03d' %(load_index))
    fig.savefig('circ_vs_lin_vertical.jpg', dpi = 400)