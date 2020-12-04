"""
Created on Fri Nov 13 09:40:18 2020

@author: Manuel
@description: Functions for post processing
"""
import matplotlib.pyplot as plt # for plotting
import numpy as np              # for array operations
import os                       # for file paths
import cv2                      # for image reading
import scipy.signal as sci 

def calc_qfield(x, y, u, v):
    # calculate the derivatives
    ux = np.gradient(u, x[0,:], axis = 1)
    uy = np.gradient(u, y[:,0], axis = 0)
    vx = np.gradient(v, x[0,:], axis = 1)
    vy = np.gradient(v, y[:,0], axis = 0)
    qfield = -0.5*(ux**2+2*vx*uy+vy**2)
    return qfield



def pad(x, y, u, v, width):
    """
    Function to pad the velocity field with 0s at the edges

    Parameters
    ----------
    x : 2d np.array
        Horizontal coordinates of the channel in pixels.
    v : 2d np.array
        Vertical velocity of the channel.
    width : int
        Width of the channel in pixels.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    """
    # set up a dummy with 0s of the same height as the velocity field
    pad_0 = np.zeros((x.shape[0],1))
    # set up a dummy with Img_Width of the same height as the velocity field
    pad_max = np.ones((x.shape[0],1))*width
    # pad the x coordinates
    x = np.hstack((pad_0, x, pad_max))
    # expand the y array by two rows
    y = np.hstack((y, y[:,:2]))
    # pad the horizontal velocity
    u = np.hstack((pad_0, u, pad_0))
    # pad the vertical velocity
    v = np.hstack((pad_0, v, pad_0))
    # return the result
    return x, y, u, v

def calc_flux(x, v):
    """
    Function to calculate the flux from the velocity field. Careful, that x and
    v are padded, meaning we have added v = 0 at the boundary conditions.

    Parameters
    ----------
    x : 1d np.array 
        Horizontal coordinates of the channel in pixels.
    v : 1d np.array
        Padded vertical velocity across the width of the channel.

    Returns
    -------
    q : int
        Flux across the width of the channel.

    """
    # calculate the flux of the padded fields
    q = np.trapz(v, x)
    # return the result
    return q

def shift_grid(x, y):
    """
    Function to shift the grid by half the window size for pcolormesh

    Parameters
    ----------
    x : 2d np,array 
        Horizontal coordinates of the interrogation window centres.
    y : 2d np.array
        Vertical coordinates of the interrogation window centres.

    Returns
    -------
    x : 2d np,array 
        Horizontal coordinates of the interrogation window edges.
    y : 2d np.array
        Vertical coordinates of the interrogation window edges.

    """
    # calculate half the width and height of the windows
    half_width = x[0,1] - x[0,0]
    half_height = y[0,0] - y[1,0]
    # shift the data so the bottom left corner is at (0,0)
    x = x-half_width
    y = y-half_height
    # stack another row onto the x values and another column onto the y values
    x = np.vstack((x, x[0,:]))
    y = np.hstack((y, np.expand_dims(y[:,0], axis = 1)))
    # expand the arrays along the other axis by a value that is larger than half the interrogation height and width
    x = np.hstack((x, np.ones((x.shape[0],1))*(np.max(x)+half_width)))
    y = np.vstack((np.ones((1,y.shape[1]))*(np.max(y)+half_height),y))
    # return the result
    return x, y
    
def set_plot_parameters(SizeLarge, SizeMedium, SizeSmall):
    """
    Function to set default plot parameters in case they were set differently
    in the rc params file
    Parameters
    ----------
    SizeLarge : int
        Fontsize for large texts.
    SizeMedium : int
        Fontsize for medium texts.
    SizeSmall : int
        Fontsize for small texts.
    """
    plt.rc('font', size=SizeMedium)         # controls default text sizes
    plt.rc('axes', titlesize=SizeLarge)     # fontsize of the axes title
    plt.rc('axes', labelsize=SizeLarge)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SizeMedium)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SizeMedium)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SizeSmall)    # legend fontsize
    plt.rc('figure', titlesize=SizeLarge)   # fontsize of the figure title
    plt.rc('font', family='serif')          # serif as text font
    plt.rc('text', usetex=True)             # enable latex
    plt.rc('')
    return

def read_lvm(path):
    """
    Function to read the experimental Labview Files.
    This file was created by Domenico

    Parameters
    ----------
    path : string
        Folder indicating the location ot the files.

    Returns
    -------
    list
        Time and pressure signal after opening the valve.

    """
    header    = 12
    value     = 15
    with open(path) as alldata:
        line = alldata.readlines()[14]
    n_samples = int(line.strip().split('\t')[1])
    time    = []
    voltage = []
    for i in range(value):
        with open(path) as alldata:                       #read the data points with the context manager
            lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
        time_temp       = [float(line.strip().split('\t')[0].replace(',','.')) for line in lines] 
        voltage_temp    = [float(line.strip().split('\t')[1].replace(',','.')) for line in lines]
        time            = np.append(time,time_temp)
        voltage         = np.append(voltage,voltage_temp)
    pressure = voltage*208.73543056621196-11.817265775905382
    return [time, pressure]

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
    # navigate to the txt data
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

def get_img_shape(Fol_Raw):
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
    img = cv2.imread(name,0)
    # take the image width
    height, width = img.shape 
    # return the width in pixels
    return height, width

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
    # check if the folder exists and create if it doesn't
    if not os.path.exists(Fol_In):
        os.makedirs(Fol_In)
    # return the name as a string
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

def separate_name(name):
    """
    Function to separate the name and put commas in between

    Parameters
    ----------
    name : str
        Name of the run without the prefix and suffix.

    Returns
    -------
    name : str
        Name of the run without the prefix and suffix separated by commas.

    """
    name = name.replace('_', ', ')
    return name

def cut_processed_name(name):
    """
    Function to cute the Results_ part and the window size

    Parameters
    ----------
    name : str
        Name of the folder with the processed data.

    Returns
    -------
    name : str
        Name of the run without the prefix and suffix.

    """
    name = name[8:-6]
    return name

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

Fol_In = 'C:\PIV_Processed\Images_Processed\Results_F_h2_f1000_1_q_24_24'
NX = get_column_amount(Fol_In)
x, y, u, v, ratio, mask = load_txt(Fol_In, 311, NX)
x, y, u, v = pad(x, y, u, v, 273)
# qfield = calc_qfield(x, y, u, v)

idx = 9
plt.plot(y[:,0], u[:,idx])
fil = sci.firwin(y.shape[0]//20, 0.0000000005, window='hamming', fs = 2)

u_filt =sci.filtfilt(b = fil, a = [1], x = u, axis = 0, padlen = 3, padtype = 'constant')
v_filt =sci.filtfilt(b = fil, a = [1], x = v, axis = 0, padlen = 3, padtype = 'constant')
plt.plot(y[:,0], u_filt[:,idx])
# fig, ax = plt.subplots()
# ax.contourf(qfield)
# ax.set_aspect(1)