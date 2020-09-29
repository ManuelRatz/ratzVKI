import numpy as np
import smooth_n as smo
import matplotlib.pyplot as plt
import os

plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=20)   # fontsize of the figure title
plt.rc('text', usetex=True)      # use latex for the text
plt.rc('font', family='serif')   # serif as text font
plt.rc('axes', grid=True)        # enable the grid
plt.rc('savefig', dpi = 100)     # set the dpi for saving figures


# load the data that is supposed to be smoothed
FOL_IN = '..' + os.sep + 'experimental_data' + os.sep + '1500_pascal' + os.sep
h_load = np.genfromtxt(FOL_IN + 'avg_height_1500.txt')
h_smoothed = smo.smoothn(h_load, s = 2.5)[0]
h_smoothed_heavy = smo.smoothn(h_load, s = 1000)[0]

# set up the time steps
t = np.arange(0, len(h_load)/500, 1/500)

# set up the folder in which to save the images
FOL_OUT = 'smoothn_images' + os.sep

"""
Here the investigation of the smoothing options for the height is done.
First the smoothed height vs raw height is compared
Afterwards the velocity is again smoothed so in total there are 4 options for
the velocity
"""

# plot the height
fig, ax = plt.subplots(figsize = (8, 5))
plt.plot(t, h_load*1000, label = 'Raw height')
plt.plot(t, h_smoothed*1000, label = 's = 2.5')
plt.plot(t, h_smoothed_heavy*1000, label = 's = 1000')
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$h$[mm]')
plt.legend()
plt.title('Height comparison for different smoothing options')
plt.savefig(FOL_OUT + 'Height_comparison_smoothing_1500.png')

# plot the height zoomed in on the beginning
fig, ax = plt.subplots(figsize = (8,5))
plt.plot(t, h_load*1000, label = 'Raw height')
plt.plot(t, h_smoothed*1000, label = 's = 2.5')
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$h$[mm]')
ax.set_xlim([-0.001, 0.01])
ax.set_ylim([77, 84])
plt.legend()
plt.title('Height comparison for different smoothing options (zoomed)')
plt.savefig(FOL_OUT + 'Height_comparison_smoothing_zoomed_1500.png')

# calculate the velocities
u_load = np.gradient(h_load)/0.002
u_raw_h_smooth = np.gradient(h_smoothed)/0.002

# smooth the velocities
u_smooth_h_raw = smo.smoothn(u_load, s = 150)[0]
u_smooth_h_smooth = smo.smoothn(u_raw_h_smooth, s = 150)[0]

# plot and compare the velocities
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Velocity comparison for different smoothing options')
plt.plot(t, u_load, label = 'Raw h, raw u')
plt.plot(t, u_raw_h_smooth, label = 'Smooth h, raw u')
plt.plot(t, u_smooth_h_raw, label = 'Raw h, smooth u')
plt.plot(t, u_smooth_h_smooth, label = 'Smooth h, smooth u')
ax.set_xlim([-0.05, 2.5])
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$u$[m/s]')
plt.legend()
plt.savefig(FOL_OUT + 'Velocity_comparison_smoothing_1500.png')

# plot and compare the velocities zoomed in on a peak
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Velocity comparison for different smoothing options (zoomed)')
plt.plot(t, u_load, label = 'Raw h, raw u')
plt.plot(t, u_raw_h_smooth, label = 'Smooth h, raw u')
plt.plot(t, u_smooth_h_raw, label = 'Raw h, smooth u')
plt.plot(t, u_smooth_h_smooth, label = 'Smooth h, smooth u')
ax.set_xlim([1.1, 1.4])
ax.set_ylim([0, 0.06])
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$u$[m/s]')
plt.legend(loc = 'upper left')
plt.savefig(FOL_OUT + 'Velocity_comparison_smoothing_zoomed_1500.png')

"""
Now we look at different smoothing options for the contact angle
"""

# load the data
ca_l = np.genfromtxt(FOL_IN + 'lca_1500.txt')
ca_r = np.genfromtxt(FOL_IN + 'rca_1500.txt')

# calculate the average contact angle in degrees
ca_load = (ca_l + ca_r) * 90 / np.pi

ca_smooth_heavy = smo.smoothn(ca_load, s = 200)[0]
ca_smooth_light = smo.smoothn(ca_load, s = 1)[0]
ca_smooth_very_heavy = smo.smoothn(ca_load, s = 1000)[0]
# # weighting array
# weight = np.ones(2001)
# weight[1250:1350] = 0.1
# ca_smooth_weight = smo.smoothn(ca_load, weight, s = 1)

# plot the differently smoothed contact angles
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Contact angle comparison for different smoothing options')
plt.plot(t, ca_load, label = 'Unfiltered contact angle')
plt.plot(t, ca_smooth_very_heavy, label = 'Very heavy smoothing')
plt.plot(t, ca_smooth_heavy, label = 'Heavy smoothing')
plt.plot(t, ca_smooth_light, label = 'Light smoothing')
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$\Theta[^\circ]$')
plt.legend(loc = 'upper right')
plt.savefig(FOL_OUT + 'Contact_angle_comparison_smoothing_1500.png')

# plot the differently smoothed contact angles zoomed in on a peak
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Contact angle comparison for different smoothing options (zoomed)')
plt.plot(t, ca_load, label = 'Unfiltered contact angle')
plt.plot(t, ca_smooth_very_heavy, label = 'Very heavy smoothing')
plt.plot(t, ca_smooth_heavy, label = 'Heavy smoothing')
plt.plot(t, ca_smooth_light, label = 'Light smoothing')
ax.set_xlabel('$t$[s]')
ax.set_ylabel('$\Theta[^\circ]$')
ax.set_xlim([2.6, 3.2])
plt.legend()
plt.savefig(FOL_OUT + 'Contact_angle_comparison_smoothing_zoomed_1500.png')































