"""
@author: ratz
@description: comparison of different smoothing options for firwin and filtfilt
for the contact angle, height and pressure measured in the experiment
"""

import numpy as np               # for numerical calculations with arrays
import scipy.signal as sci       # for the frequency based filtering
import matplotlib.pyplot as plt  # for plotting things
import os                        # for the creation of data paths

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

fs = 500 # this is the sampling rate which equals the fps

# iterate over all the signals
for i in range(0, 3):
    # set up the input and output folder
    Fol_In = '..' + os.sep + 'experimental_data' + os.sep + '%d_pascal' %pressure_array[i] + os.sep
    Fol_Out = 'filtfilt_images' + os.sep + '%d_pa' %pressure_array[i] + os.sep
    
    # load the height data
    height = np.genfromtxt(Fol_In + 'avg_height_%d.txt' %pressure_array[i])
    
    # set up the timesteps
    t = np.arange(0, len(height)/fs, 1/fs)
    
    """
    First we take a look at the height signals
    """    
    
    # define the kernel as 1/10 of the total length
    n0 = int(len(height)/10)
    # set up the filtering windows with different cutoff frequencies
    fil = sci.firwin(n0, 2/fs*2 , window='hamming')
    height_filt1 = sci.filtfilt(b = fil, a = [1], x = height)
    fil2 = sci.firwin(n0, 25/fs*2 , window='hamming')
    height_filt2 = sci.filtfilt(b = fil2, a = [1], x = height)

    create_figure('$h$[mm]')
    plt.plot(t, height*1000, label = 'Raw height')
    plt.plot(t, height_filt1*1000, label = 'Filtered height, f = 2Hz')
    plt.plot(t, height_filt2*1000, label = 'Filtered height, f = 25Hz')
    plt.legend()
    plt.savefig(Fol_Out + 'Height_comparison_filtfilt_%d.png' %pressure_array[i])

    create_figure('$h$[mm]', [0, 0.01], [height[0]*1000, height[0]*1000+4])
    plt.plot(t, height*1000, label = 'Raw height')
    plt.plot(t, height_filt1*1000, label = 'Filtered height, f = 2Hz')
    plt.plot(t, height_filt2*1000, label = 'Filtered height, f = 25Hz')
    plt.legend()
    plt.savefig(Fol_Out + 'Height_comparison_filtfilt_zoomed_%d.png' %pressure_array[i])
    
    """
    Now the contact angle
    """
    # load the data
    ca_l = np.genfromtxt(Fol_In + 'rca_%d.txt' %pressure_array[i])*180/np.pi
    
    # set up the filtering windows for different cutoff frequencies and filter
    fil = sci.firwin(n0, 2/fs*2 , window='hamming')
    ca_filt1 = sci.filtfilt(b = fil, a = [1], x = ca_l)
    fil2 = sci.firwin(n0, 10/fs*2 , window='hamming')
    ca_filt2 = sci.filtfilt(b = fil2, a = [1], x = ca_l)
    fil3 = sci.firwin(n0, 5/fs*2 , window='hamming')
    ca_filt3 = sci.filtfilt(b = fil3, a = [1], x = ca_l)
    
    # plot the global scale
    create_figure('$\Theta[^\circ]$', ylim = [25, 110])
    plt.plot(t, ca_l, label = 'Raw contact angle')
    plt.plot(t, ca_filt1, label = 'Filtered, f = 2Hz')
    plt.plot(t, ca_filt3, label = 'Filtered, f = 5Hz')
    plt.plot(t, ca_filt2, label = 'Filtered, f = 10Hz')
    plt.legend(ncol = 2)
    plt.savefig(Fol_Out + 'Contact_angle_comparison_filtfilt_%d.png' %pressure_array[i])
    
    # plot a zoomed peak
    create_figure('$\Theta[^\circ]$', [2.6, 3.5], [25, 80])
    plt.plot(t, ca_l, label = 'Raw contact angle')
    plt.plot(t, ca_filt1, label = 'Filtered, f = 2Hz')
    plt.plot(t, ca_filt3, label = 'Filtered, f = 5Hz')
    plt.plot(t, ca_filt2, label = 'Filtered, f = 10Hz')
    plt.legend()
    plt.savefig(Fol_Out + 'Contact_angle_comparison_filtfilt_zoomed_%d.png' %pressure_array[i])
    
    """
    Finally we look at the pressure signals
    """
    #load the pressure signal
    pres = np.genfromtxt(Fol_In + 'pressure_%d.txt' %pressure_array[i])
    
    # set up the filtering windows for different cutoff frequencies
    fil = sci.firwin(n0, 2/fs*2 , window='hamming')
    pres_filt1 = sci.filtfilt(b = fil, a = [1], x = pres)
    fil2 = sci.firwin(n0, 10/fs*2 , window='hamming')
    pres_filt2 = sci.filtfilt(b = fil2, a = [1], x = pres)
    
    create_figure('$p$[Pa]', [0, 3.5], [615+i*175, 665+i*175])
    if i == 2: # only plot the unfiltered pressure for 1500 pa
        plt.plot(t, pres, label = 'Raw pressure')
    for j in range(1, 6, 1):
        fil_temp = sci.firwin(n0, j/fs*2 , window='hamming')
        pres_filt_temp = sci.filtfilt(b = fil_temp, a = [1], x = pres)
        plt.plot(t, pres_filt_temp, label = 'Filtered, f =%dHz' %j)
    plt.legend(ncol = 2)
    plt.savefig(Fol_Out + 'Pressure_comparison_filtfilt_%d.png' %pressure_array[i])
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, pres, label = 'Raw pressure')
    plt.plot(t, pres_filt1, label = 'Filtered, f = 2Hz')
    plt.plot(t, pres_filt2, label = 'Filtered, f = 10Hz')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$p$[Pa]')
    ax.set_xlim([0, 4])
    # ax.set_ylim(950,1025)
    plt.legend()
    # plt.title('Height comparison for different smoothing options')
    plt.savefig(Fol_Out + 'Pressure_comparison_filtfilt_zoomed_%d.png' %pressure_array[i])