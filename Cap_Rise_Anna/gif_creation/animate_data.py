"""
@author: ratz
@description:
This code creates animations of 4 data sets that are taken from the experiment
in the narrow channel for the dynamic contact angle. We extract the data from
the hard disk and plot them for each new time step. Finally we append everything
into a gif to create an animation
"""

"""
Right now this is just for testing stuff, no actual data is being used (02.10.)
"""

import numpy as np               # this is for array operations
import matplotlib.pyplot as plt  # this is for plots
import os                        # this is for setting up data paths
import imageio                   # this is for the animation
from scipy.signal import savgol_filter # this is for filtering signals

# Settings for the plots
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

# this function creates a plot up until the current index for the gifs
def create_plot_of_index(y_values1, label1, Fol_Out, abbrev, index, y_low, y_high, y_values2 = None, label2 = None):
    """
    This function creates a plot of up to 4 data sets until the current index
    Additionally there the leading point at the index is shown with a scatter plot
    The user is updated every 50 plots in case there are large data sets to see
    how the calculation is proceeding
    
    Parameters
    ----------
    y_values1 : 1d Array of length len(t)
        Values of the y-axis for the plot.
    Fol_Out : string
        Name of the output folder and also naming convention for the image.
    abbrev: string
        abbreviation of the image name.
    index : integer
        Index of the current index of the array, also added in the title.
    y_low : float or integer
        Lower bound of the y-values.
    y_high : float or integer
        Upper bound of the y-values.
    y_values2 : 1d Array of length len(t), optional
        Values of the second data set for the plot. The default is None.

    Returns
    -------
    Images of the plots saved as pngs.

    """
    # take every fourth image
    index = index*3
    # create the figure
    fig, ax = plt.subplots(figsize = (4.5, 3))
    ax.set_xlim([0, 4]) # limit of the x-axis
    ax.set_ylim([y_low, y_high]) # limit of the y-axis
    # plot the first data set
    ax.plot(t[:index+1], y_values1[:index+1], label = label1) # plot the first data set
    ax.scatter(t[index], y_values1[index]) # plot the leading point of the first data set
    if y_values2 is not None: # check whether y_values2 was given
        ax.plot(t[:index+1], y_values2[:index+1], label = label2) # plot the second data set
        ax.scatter(t[index], y_values2[index]) # plot the leading point of the second data set
    # set the title with the current image, save and close the plot
    plt.title('Image %04d' % (index+1)) # set the title as the number of the current iteration
    plt.legend(loc = 'upper right') # set up the legend
    plt.savefig(Fol_Out + os.sep + abbrev + '_Img%04d.png' %(index+1)) # save the figure
    plt.close(fig) # close the figure to not overcrowd the plot window
 
def create_folder(folder):
    """
    Function to create a folder and return the name as a string
    
    Parameters
    ----------
    folder : string
        Name of the folder that is to be created.

    Returns
    -------
    folder : string
        Name of the created folder.

    """
    path = '..' + os.sep + '..' + os.sep + '..' + os.sep + 'DATA' + os.sep + 'Images_Anna' + os.sep + '1500_pa' + os.sep + folder # set up the saving path
    if not os.path.exists(path): # create the folder if it does not exist
        os.mkdir(path)
    return path
 
def create_gif(Folder, abbrev, im_duration):
    GIFNAME = Folder + '.gif' # name of the gif
    images = [] # empty list to append the images
    for k in range(0, LENGHT):
        FIG_NAME = Folder + os.sep + abbrev + '_Img%04d.png' % ((3*k) +1) # name of the image
        images.append(imageio.imread(FIG_NAME)) # append into the image folder
        # update the user on the status every 50 images
        if (((k+1) % 50) == 0):
            print('Finished the first %d images of %s' %(k+1, abbrev))
    imageio.mimsave(GIFNAME, images, duration = im_duration) # save the gif


# load the contact angle and height
Fol_In = '..' + os.sep + 'experimental_data' + os.sep + '1500_pascal' + os.sep
lca = np.genfromtxt(Fol_In + 'lca_1500.txt')
rca = np.genfromtxt(Fol_In + 'rca_1500.txt')
height = np.genfromtxt(Fol_In + 'avg_height_1500.txt')

# smooth the data
lca_smooth = savgol_filter(lca*180/np.pi, 9, 3, axis = 0) # smooth the left ca
rca_smooth = savgol_filter(rca*180/np.pi, 9, 3, axis = 0) # smooth the right ca
height_smooth = savgol_filter(height, 9, 2, axis = 0) # smooth the height
vel = np.gradient(height_smooth)/0.002 # calculate the velocity
vel_smooth = savgol_filter(vel, 55, 1, axis = 0) # smooth the velocity
acc = np.gradient(vel_smooth)/0.002 # calculate the acceleration
acc_smooth = savgol_filter(acc, 75, 1, axis = 0) # smooth the acceleration

# Give the constants
mu     = 8.90*10**(-4)  # dynamic viscosity (Pa s)
sigma  = 0.07197        # surface tension (N/m)
delta  = 0.005          # channel width (m)
rhoL   = 1000           # density (kg/m^3)

# Calculate the Weber number and Capillary number
We = rhoL*vel_smooth**2*delta/sigma
Ca = mu*vel_smooth/sigma

LENGHT = 666  # amount of plots to be made and put into the gif

# set up the timesteps
t = np.arange(0, len(lca_smooth)/500, 1/500)

# create the folders to put in the plots
Fol_ca = create_folder('ca')    # the contact angle
Fol_vel = create_folder('vel')  # the velocity
Fol_acc = create_folder('acc')  # the acceleration
Fol_we = create_folder('We')    # the Weber number (mu*u^2*rho/sigma)
Fol_Ca = create_folder('Ca')    # the Capillary number (u*mu/sigma)


# create a plot for each timestep by calling the function LENGTH times
for k in range(0, LENGHT):
    # simply comment the respective line to calculate/not calculate a plot
    create_plot_of_index(lca_smooth, '$\Theta_l$', Fol_ca, 'ca', k, 25, 120, rca_smooth, '$\Theta_r$')
    create_plot_of_index(vel_smooth, 'Smoothed', Fol_vel, 'vel', k, -0.1, 0.4)
    create_plot_of_index(acc_smooth, 'Smoothed', Fol_acc, 'acc', k, -2, 0.7)
    create_plot_of_index(Ca, '$Ca$', Fol_Ca, 'Ca', k, -0.0014, 0.005)
    create_plot_of_index(We, '$We$', Fol_we, 'We', k, 0, 2)
    # update the user on the status every 50 images
    if (((k+1) % 50) == 0):
        print('Finished the first %d plots' %(k+1))

# create the gifs, comment the respective line to disable/enable gif creation
create_gif(Fol_ca, 'ca', 0.01)
create_gif(Fol_vel, 'vel', 0.01)
create_gif(Fol_acc, 'acc', 0.01)
create_gif(Fol_Ca, 'Ca', 0.01)
create_gif(Fol_we, 'We', 0.01)













