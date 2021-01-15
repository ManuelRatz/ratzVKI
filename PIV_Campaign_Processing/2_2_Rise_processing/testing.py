'''
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
'''

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import post_processing_functions as ppf
fontsize = 20
fontProperties = {'family':'sans-serif', 'size' : fontsize}

""" histogram of a typical SNR distribution """

Fol = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h3_f1200_1_p14_64_16'
NX = ppf.get_column_amount(Fol)
Fol_Raw = ppf.get_raw_folder(Fol)
idx = 300
raw_img = ppf.load_raw_image(Fol_Raw, idx)
x, y, u, v, ratio, valid = ppf.load_txt(Fol, idx, NX)
valid = valid.astype(bool)
invalid = ~valid

fig = plt.figure(figsize = (6,5))
ax = fig.gca()
ax.hist(ratio.ravel(), bins = 100, density = True)
ax.set_xlim(0, 20)
ax.set_xlabel('SNR[-]', fontsize = fontsize+2)
ax.set_ylabel('Probability [\%]', fontsize = fontsize+2)
ax.plot([6.5,6.5],[0,100], lw = 2, color = 'red', label = 'Threshold')
ax.legend(prop={'size': fontsize}, loc = 'upper right')
ax.set_ylim(0, 0.25)
ax.set_xticklabels(np.linspace(0, 20, 5, dtype = int), fontProperties)
ax.set_yticklabels(np.arange(0, 25, 5, dtype = int), fontProperties)
fig.tight_layout(pad = 0)
fig.savefig('histogram.png', dpi = 150)

fig = plt.figure(figsize = (2, 5), frameon = False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(np.flipud(raw_img), cmap = plt.cm.gray)
fig.savefig('raw_im_for_hist.png', dpi = 150)


fig = plt.figure(figsize = (2, 5))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.set_ylim(0, 1270)
ax.quiver(x[valid], y[valid], u[valid], v[valid], scale = 100, color = 'red')
ax.quiver(x[invalid], y[invalid], u[invalid], v[invalid], scale = 100, color = 'blue')
ax.set_aspect(1)
fig.savefig('quiv_for_hist.png', dpi = 150)


#%%
""" 3D plots of the correlation maps for the report """


x = np.arange(0, 16, 1)+0.5
y = np.arange(0, 64, 1)+0.5
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()


corr = np.load('corr_map_valid.npy')
corr_plot = corr.flatten()

fig = plt.figure(figsize = (8, 5))
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, corr_plot, linewidth=0.2, cmap = plt.cm.viridis)
ax.set_xticklabels(ax.get_xticks(), fontProperties)
ax.set_yticklabels(ax.get_yticks(), fontProperties)
ax.set_zticklabels(ax.get_zticks(), fontProperties)
ax.set_zlim(0, 1)
ax.set_xlim(0, 16)
ax.set_ylim(0, 64)
ax.set_xticks(np.linspace(0, 16, 5))
ax.set_yticks(np.linspace(0, 64, 5))
ax.set_xlabel('$x$[px]', labelpad = 15)
ax.set_ylabel('$y$[px]', labelpad = 16)
ax.set_zlabel('Correlation[\%]', labelpad = 12)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_zticklabels(np.arange(0, 101, 20), fontsize = fontsize)
ax.set_xticklabels(np.arange(0, 17, 4))
ax.set_yticklabels(np.arange(0, 65, 16))
fig.tight_layout(pad = 2)
fig.savefig('map_valid.png', dpi = 150)
print(np.max(corr)/np.std(corr))

corr = np.load('corr_map_invalid.npy')
corr_plot = corr.flatten()

fig = plt.figure(figsize = (8, 5))
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, corr_plot, linewidth=0.2, cmap = plt.cm.viridis)
ax.set_xticklabels(ax.get_xticks(), fontProperties)
ax.set_yticklabels(ax.get_yticks(), fontProperties)
ax.set_zticklabels(ax.get_zticks(), fontProperties)
ax.set_zlim(0, 1)
ax.set_xlim(0, 16)
ax.set_ylim(0, 64)
ax.set_xticks(np.linspace(0, 16, 5))
ax.set_yticks(np.linspace(0, 64, 5))
ax.set_xlabel('$x$[px]', labelpad = 15)
ax.set_ylabel('$y$[px]', labelpad = 16)
ax.set_zlabel('Correlation[\%]', labelpad = 12)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_zticklabels(np.arange(0, 101, 20), fontsize = fontsize)
ax.set_xticklabels(np.arange(0, 17, 4))
ax.set_yticklabels(np.arange(0, 65, 16))
fig.tight_layout(pad = 2)
fig.savefig('map_invalid.png', dpi = 150)
print(np.max(corr)/np.std(corr))

