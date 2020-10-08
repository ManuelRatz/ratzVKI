"""
@author: ratz
@description: comparison of different smoothing options for the smoothn function
for the contact angle, height and pressure measured in the experiment
"""

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

def create_figure(ylabel, xlim = None, ylim = None, xlabel = None):
    """
    Parameters
    ----------
    ylabel : string
        Label of the y-axis.
    xlim : tuple with two entries, optional
        Lower and upper limit for the x-axis, default is [0, 4]
    ylim : tuple with two entries, optional
        Lower and upper limit for the y-axis.
    xlabel : string, optional
        Label of the x-axis, default is '$t$[s]'
    Returns
    -------
    The created figure to put plots in.
    """
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        ax.set_xlim(0, 4)
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('$t[s]$')
    return

# define the array with the 3 different pressures for naming purposes
pressure_array=np.array([1000, 1250, 1500]) 

# iterate over all three pressures
for i in range(0, 3):
    # load the data that is supposed to be smoothed
    FOL_IN = '..' + os.sep + 'experimental_data' + os.sep + '%d_pascal' %pressure_array[i] + os.sep
    height = np.genfromtxt(FOL_IN + 'avg_height_%d.txt' %pressure_array[i])
    h_smoothed = smo.smoothn(height, s = 2.5)[0]
    h_smoothed_heavy = smo.smoothn(height, s = 500)[0]
    
    # set up the time steps
    t = np.arange(0, len(height)/500, 1/500)
    
    # set up the folder in which to save the images
    FOL_OUT = 'smoothn_images' + os.sep + '%d_pa' %pressure_array[i] + os.sep
    
    """
    Here the investigation of the smoothing options for the height is done.
    First the smoothed height vs raw height is compared
    Afterwards the velocity is again smoothed so in total there are 4 options for
    the velocity
    """
    
    # plot the height
    create_figure('$h$[mm]')
    plt.plot(t, height*1000, label = 'Raw height')
    plt.plot(t, h_smoothed*1000, label = 's = 2.5')
    plt.plot(t, h_smoothed_heavy*1000, label = 's = 500')
    plt.legend()
    plt.savefig(FOL_OUT + 'Height_comparison_smoothn_%d.png' %pressure_array[i])
    
    # plot the height zoomed in on the beginning
    create_figure('$h$[mm]', [0, 0.01], [height[0]*1000, height[0]*1000+4])
    plt.plot(t, height*1000, label = 'Raw height')
    plt.plot(t, h_smoothed*1000, label = 's = 2.5')
    plt.plot(t, h_smoothed_heavy*1000, label = 's = 500')
    plt.legend()
    plt.savefig(FOL_OUT + 'Height_comparison_smoothn_zoomed_%d.png' %pressure_array[i])
    
    # calculate the velocities, one from the raw data, one from the smoothed
    u_load = np.gradient(height)/0.002
    u_raw_h_smooth = np.gradient(h_smoothed)/0.002
    
    # smooth the velocities, once for the raw velocity and once for the already smoothed
    u_smooth_h_raw = smo.smoothn(u_load, s = 150)[0]
    u_smooth_h_smooth = smo.smoothn(u_raw_h_smooth, s = 150)[0]
    
    # plot and compare the smoothing techniques for the velocities
    create_figure('$u$[m/s]', [-0.05, 2.5])
    plt.plot(t, u_load, label = 'Raw h, raw u')
    plt.plot(t, u_raw_h_smooth, label = 'Smooth h, raw u')
    plt.plot(t, u_smooth_h_raw, label = 'Raw h, smooth u')
    plt.plot(t, u_smooth_h_smooth, label = 'Smooth h, smooth u')
    plt.legend()
    plt.savefig(FOL_OUT + 'Velocity_comparison_smoothn_%d.png' %pressure_array[i])
    
    # plot and compare the velocities zoomed in on a peak
    create_figure('$u$mm/s', [1.1, 1.6], [-0.03, 0.06])
    plt.plot(t, u_load, label = 'Raw h, raw u')
    plt.plot(t, u_raw_h_smooth, label = 'Smooth h, raw u')
    plt.plot(t, u_smooth_h_raw, label = 'Raw h, smooth u')
    plt.plot(t, u_smooth_h_smooth, label = 'Smooth h, smooth u')
    plt.legend(loc = 'upper right')
    plt.savefig(FOL_OUT + 'Velocity_comparison_smoothn_zoomed_%d.png' %pressure_array[i])
    
    # calculate and plot the velocities locally for different smoothing settings
    u_low = smo.smoothn(u_load, s = 25)[0]
    u_med = smo.smoothn(u_load, s = 100)[0]
    u_high = smo.smoothn(u_load, s = 400)[0]
    u_ext = smo.smoothn(u_load, s = 1000)[0]
    create_figure('$u$[m/s]', [1.1, 2.3], [-0.03, 0.06])
    plt.plot(t, u_load, label = 'Unsmoothed')
    plt.plot(t, u_low, label = 's = 25')
    plt.plot(t, u_med, label = 's = 100')
    plt.plot(t, u_high, label = 's = 400')
    plt.plot(t, u_ext, label = 's = 1000')
    plt.legend(loc = 'upper right', ncol = 2)
    plt.savefig(FOL_OUT + 'Velocity_comparison_s_smoothn_%d.png' %pressure_array[i])
    
    """
    Now we look at different smoothing options for the contact angle
    """
    
    # load the data
    ca_r = np.genfromtxt(FOL_IN + 'rca_%d.txt' %pressure_array[i])
    
    # calculate the average contact angle in degrees
    ca_load = ca_r * 180 / np.pi
    
    ca_smooth_heavy = smo.smoothn(ca_load, s = 200)[0]
    ca_smooth_light = smo.smoothn(ca_load, s = 1)[0]
    ca_smooth_very_heavy = smo.smoothn(ca_load, s = 1000)[0]
    
    # plot the differently smoothed contact angles
    create_figure('$\Theta[^\circ]$')
    plt.plot(t, ca_load, label = 'Unfiltered contact angle')
    plt.plot(t, ca_smooth_very_heavy, label = 'Very heavy smoothing')
    plt.plot(t, ca_smooth_heavy, label = 'Heavy smoothing')
    plt.plot(t, ca_smooth_light, label = 'Light smoothing')
    plt.legend(loc = 'upper right')
    plt.savefig(FOL_OUT + 'Contact_angle_comparison_smoothn_%d.png' %pressure_array[i])
    
    # plot the differently smoothed contact angles zoomed in on a peak
    create_figure('$\Theta[^\circ]$', xlim = [2.6, 3.2])
    plt.plot(t, ca_load, label = 'Unfiltered contact angle')
    plt.plot(t, ca_smooth_very_heavy, label = 's = 1000')
    plt.plot(t, ca_smooth_heavy, label = 's = 200')
    plt.plot(t, ca_smooth_light, label = 's = 1')
    plt.legend()
    plt.savefig(FOL_OUT + 'Contact_angle_comparison_smoothn_zoomed_%d.png' %pressure_array[i])

    """
    And finally different smoothing options for the pressure
    """
    # load the data
    pres = np.genfromtxt(FOL_IN + 'pressure_%d.txt' %pressure_array[i])
    
    # smooth the data
    pres_smo_light = smo.smoothn(pres, s = 1)[0]
    pres_smo_heavy = smo.smoothn(pres, s = 200)[0]
    
    create_figure('$p$[Pa]')
    plt.plot(t, pres, label = 'Unfiltered pressure')
    plt.plot(t, pres_smo_light, label = 's = 1')
    plt.plot(t, pres_smo_heavy, label = 's = 200')
    plt.legend(loc = 'lower right')
    plt.savefig(FOL_OUT + 'Pressure_comparison_smoothn_%d.png' %pressure_array[i])
    
    




























