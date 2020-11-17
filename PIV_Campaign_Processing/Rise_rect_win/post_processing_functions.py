"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Functions for post processing
"""
import matplotlib.pyplot as plt # for plotting
import numpy as np              # for array operations
import os                       # for file paths

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
    Function to load a txt and reshape the fieilds.

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
        Array containg the x coordinates of the interrogation window centres.
        DESCRIPTION.
    y : 2d np.array
        Array containg the y coordinates of the interrogation window centres.
    u : 2d np.array
        Array containing the u component for every interrogation window.
    v : 2d np.array
        Array containing the v component for every interrogation window.

    """
    # set the file name
    file_name = Fol_In + os.sep + 'field_A%06d.txt' % idx
    # load the data
    data = np.genfromtxt(file_name)
    # get the amount of points in the field
    nxny = data.shape[0]
    # calculate the number of rows
    ny = nxny // nx
    # calculate the components of the velocity field and the positions by reshaping
    x, y, u, v = data[:, 0].reshape((ny, nx)), data[:, 1].reshape((ny, nx)),\
        data[:, 2].reshape((ny, nx)), data[:, 3].reshape((ny, nx))
    # return the arrays
    return  x, y, u, v

# Fol_In = 'C:\\Users\manue\Desktop\Results_12_48_R_h1_f1200_1_p15'
# nx = get_column_amount(Fol_In)
# x, y, u, v = load_txt(Fol_In, 1, nx)