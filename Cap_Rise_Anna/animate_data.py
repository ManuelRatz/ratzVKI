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

import numpy as np               # this is for mathematical stuff
import matplotlib.pyplot as plt  # this is used for plots
from PIL import Image            # this is for loading the pngs
import os                        # this is for setting up data paths
import imageio                   # this is used for the animation

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
def create_plot_of_index(y_values1, Fol_Out, abbrev, index, y_low, y_high, y_values2 = 0, y_values3 = 0, y_values4 = 0):
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
        Values of the second data set for the plot. The default is 0.
    y_values3 : 1d Array of length len(t), optional
        Values of the third data set for the plot. The default is 0.
    y_values4 : 1d Array of length len(t), optional
        Values of the fourth data set for the plot. The default is 0.

    Returns
    -------
    Images of the plots saved as pngs.

    """
    # create the figure
    fig, ax = plt.subplots(figsize = (4.5, 3))
    ax.set_xlim([0, 4])
    ax.set_ylim([y_low, y_high])
    # plot the first data set
    ax.plot(t[:index+1], y_values1[:index+1])
    ax.scatter(t[index], y_values1[index])
    # check whether the other data sets are 0, if not plot them as well
    if (np.sum(abs(y_values2)) > 0.01):
        ax.plot(t[:index+1], y_values2[:index+1])
        ax.scatter(t[index], y_values2[index])
    if (np.sum(abs(y_values3)) > 0.01):
        ax.plot(t[:index+1], y_values3[:index+1])
        ax.scatter(t[index], y_values3[index])
    if (np.sum(abs(y_values4)) > 0.01):
        ax.plot(t[:index+1], y_values4[:index+1])
        ax.scatter(t[index], y_values4[index])
    # set the title with the current image, save and close the plot
    plt.title('Image %04d' % (index+1))
    plt.savefig(Fol_Out + os.sep + abbrev + '_Img%04d.png' %(index+1))
    plt.close(fig)
 
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
    path = '..' + os.sep + '..' + os.sep + 'DATA' + os.sep + 'Animated_images' + os.sep + folder
    if not os.path.exists(path):
        os.mkdir(path)
    return path
 
def create_gif(Folder, abbrev, im_duration):
    GIFNAME = Folder + '.gif'
    images = []
    for k in range(0, LENGHT):
        FIG_NAME = Folder + os.sep + abbrev + '_Img%04d.png' % (k +1)
        images.append(imageio.imread(FIG_NAME))
        # update the user on the status every 50 images
        if (((k+1) % 50) == 0):
            print('Finished the first %d images of %s' %(k+1, abbrev))
    imageio.mimsave(GIFNAME, images, duration = im_duration)


LENGHT = 10  # amount of plots to be made and put into the gif

# set up the timesteps
t = np.arange(0, 4, 4/LENGHT)

Fol_Cos = create_folder('Cos')  # test folder to store the images of cos(t)
Fol_Sin = create_folder('Sin')  # test folder to store the images of sin(t)

# create a plot for each timestep by calling the function LENGTH times
for k in range(0, LENGHT):
    create_plot_of_index(np.cos(np.pi*t), Fol_Cos, 'cos', k, -1, 1)
    create_plot_of_index(np.sin(np.pi*t), Fol_Sin, 'sin', k, -1, 1)
    # update the user on the status every 50 images
    if (((k+1) % 50) == 0):
        print('Finished the first %d plots' %(k+1))

create_gif(Fol_Cos, 'cos', 0.01)
create_gif(Fol_Sin, 'sin', 0.01)













