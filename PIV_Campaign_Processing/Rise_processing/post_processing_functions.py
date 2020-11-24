"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Functions for post processing
"""
import matplotlib.pyplot as plt # for plotting
import numpy as np              # for array operations
import os                       # for file paths
import cv2                      # for image reading

def set_plot_parameters():
    """
    Function to set default plot parameters in case they were set differently
    in the rc params file
    """
    # define some plot parameters
    plt.rc('font', size=15)          # controls default text sizes
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('figure', titlesize=20)   # fontsize of the figure title
    plt.rc('font', family='serif')   # serif as text font
    plt.rc('text', usetex=True)      # enable latex

def get_column_amount(Fol_In):
    """
    Function to calculate the amount of columns from the .txt file.

    Parameters
    ----------
    Fol_In : string
        Input folder where the .txt files are located.

    Returns
    -------
    ny : int
        Amount of columns in the fields.

    """
    Fol_In = Fol_In + os.sep + 'data_files'
    # get the name of the first file
    file0 = os.listdir(Fol_In)[0]
    # set the input path
    file_name = Fol_In + os.sep + file0
    # load the data from the .txt file
    data = np.genfromtxt(file_name)
    # get y coordinates
    Y_S = data[:, 1]
    # calculate the index where they are shifting
    GRAD_Y = np.diff(Y_S)
    IND_X = np.where(GRAD_Y != 0)
    # take the first number
    DAT = IND_X[0]
    # get amount from the index by adding a 1
    return DAT[0] + 1    


def load_txt(Fol_In, idx, nx):
    """
    Function to load a txt and reshape the fields.

    Parameters
    ----------
    Fol_In : string
        Input folder where the .txt files are located.
    idx : int
        Index of the file to load.
    ny : int
        AMount of columns in the fields.

    Returns
    -------
    x : 2d np.array
        Array containing the x coordinates of the interrogation window centres.
    y : 2d np.array
        Array containing the y coordinates of the interrogation window centres.
    u : 2d np.array
        Array containing the u component for every interrogation window.
    v : 2d np.array
        Array containing the v component for every interrogation window.

    """
    Fol_In = Fol_In + os.sep + 'data_files'
    # set the file name
    file_name = Fol_In + os.sep + 'field_%06d.txt' % idx
    # load the data
    data = np.genfromtxt(file_name)
    # get the amount of points in the field
    nxny = data.shape[0]
    # calculate the number of rows
    ny = nxny // nx
    # calculate the components of the velocity field and the positions by reshaping
    x, y, u, v, sig2noise, valid = data[:, 0].reshape((ny, nx)), data[:, 1].reshape((ny, nx)),\
        data[:, 2].reshape((ny, nx)), data[:, 3].reshape((ny, nx)), data[:, 4].reshape((ny, nx)), data[:, 5].reshape((ny, nx))
    # return the arrays
    return  x, y, u, v, sig2noise, valid

def get_img_width(Fol_Raw):
    """
    Function to extract the image width in pixels.

    Parameters
    ----------
    Fol_Raw : string
        Location of the raw images.

    Returns
    -------
    width : int
        Width of the image in pixels.

    """
    # get the name of the first image in the folder
    name = Fol_Raw + os.sep +os.listdir(Fol_Raw)[0]
    # load the image
    img = cv2.imread(name)
    # take the image width
    width = img.shape[1] 
    # return the width in pixels
    return width

def get_frequency(Fol_Raw):
    """
    Function to extract the acquisition frequency from the image name.

    Parameters
    ----------
    Fol_Raw : string
        Location of the raw images.

    Returns
    -------
    frequency : int
        Acquisition frequcny in Hz.

    """
    # get the name of the first image in the folder
    name = os.listdir(Fol_Raw)[0]
    # crop the beginning indicating the height and R/F. The frequency ALWAYS comes after that
    name = name[6:]
    # get the indices of the '_' to see when frequency string stops
    indices = [i for i, a in enumerate(name) if a == '_']
    # crop to the frequency and convert to integer
    frequency = int(name[:indices[0]])
    # return the value
    return frequency
Fol =  'C:\PIV_Processed\Images_Preprocessed\R_h1_f1000_1_p13'
f = get_frequency(Fol)

def create_folder(Fol_In):
    """
    Function to create a folder and return the name.

    Parameters
    ----------
    Fol_In : string
        Location of the input folder.

    Returns
    -------
    Fol_In : string
        Location of the input folder.

    """
    if not os.path.exists(Fol_In):
        os.makedirs(Fol_In)
    return Fol_In

def load_h(Fol_In):
    """
    Function to load the h(t) that gets exported after the run is complete

    Parameters
    ----------
    Fol_In : string
        Location of the input folder.

    Returns
    -------
    h : 1d numpy array of float64
        Array containing the heights, length is the same as the number of image pairs.

    """
    h = np.genfromtxt(Fol_In + os.sep + 'interface_position.txt')
    return h

def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap 
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap