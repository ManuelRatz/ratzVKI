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
import smoothn
from scipy.ndimage import gaussian_filter

def pad(x, y, u, v, width):
    """
    Function to pad the velocity profiles with 0s at the walls

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    """
    # set up a dummy with 0s of the same height as the velocity field
    pad_0 = np.zeros((x.shape[0],1))
    # set up a dummy with Img_Width of the same height as the velocity field
    pad_max = np.ones((x.shape[0],1))*width
    # pad the x coordinates
    x_p = np.hstack((pad_0, x, pad_max))
    # expand the y array by two rows
    y_p = np.hstack((y, y[:,:2]))
    # pad the horizontal velocity
    u_p = np.hstack((pad_0, u, pad_0))
    # pad the vertical velocity
    v_p = np.hstack((pad_0, v, pad_0))
    # return the result
    return x_p, y_p, u_p, v_p

def calc_flux(x, v):
    """
    Function to calculate the flux from the velocity field. Careful, that x and
    v are padded, meaning we have added v = 0 at the boundary conditions.
    The result is not converted to mm, we integrate with pixels

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
    x : 2d np.array
        Array containing the x coordinates of the interrogation window centres.
    y : 2d np.array
        Array containing the y coordinates of the interrogation window centres.
    u : 2d np.array
        Array containing the u component for every interrogation window.
    v : 2d np.array
        Array containing the v component for every interrogation window.

    Returns
    -------
    x_p : 2d np.array
        Array containing the padded x coordinates of the interrogation window centres.
    y_p : 2d np.array
        Array containing the padded y coordinates of the interrogation window centres.
    u_p : 2d np.array
        Array containing the padded u component for every interrogation window.
    v_p : 2d np.array
        Array containing the padded v component for every interrogation window.

    """
    # calculate half the width and height of the windows
    half_width = (x[0,1] - x[0,0])/2
    half_height = (y[0,0] - y[1,0])/2
    # shift the data so the bottom left corner is at (0,0)
    x = x-half_width
    y = y-half_height
    # stack another row onto the x values and another column onto the y values
    x = np.vstack((x, x[0,:]))
    y = np.hstack((y, np.expand_dims(y[:,0], axis = 1)))
    # expand the arrays along the other axis by a value that is larger than half the interrogation height and width
    x = np.hstack((x, np.ones((x.shape[0],1))*(np.max(x)+2*half_width)))
    y = np.vstack((np.ones((1,y.shape[1]))*(np.max(y)+2*half_height),y))
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

def get_raw_folder(Fol_In):
    """
    Function to navigate to the raw folder.
    WARNING: Check the data path in the function, might be different for each user

    Parameters
    ----------
    Fol_In : string
        File path of the processed PIV data.

    Returns
    -------
    Fol_Raw : str
        File path of the raw PIV images.

    """
    # get the indices of the backslashs
    backslashs = [i for i, a in enumerate(Fol_In) if a == os.sep]
    # get the name of the results folder
    result_folder = Fol_In[backslashs[-1]+1:]
    # cut the suffix and prefix
    cut_name = cut_processed_name(result_folder)
    # create the output path
    Fol_Raw = os.path.join('C:\PIV_Processed\Images_Preprocessed', cut_name)
    # return the result
    return Fol_Raw

def load_raw_image(Fol_Raw, idx):
    """
    Function to load a raw image of the given index.

    Parameters
    ----------
    Fol_Raw : string
        File path of the raw PIV images.
    idx : int
        Index of the image that gets loaded.

    Returns
    -------
    img : 2d np.array
        Greyscale PIV image, flipped, because the origins are different for
        data and the image.

    """
    # get the indices of the backslashs
    backslashs = [i for i, a in enumerate(Fol_Raw) if a == os.sep]
    # get the name of the current run
    run = Fol_Raw[backslashs[-1]+1:]
    # create the image name
    img_name = Fol_Raw + os.sep + run + '.%06d.tif' %idx
    # load the image
    img = cv2.imread(img_name, 0)
    # flip it
    img = np.flip(img, axis = 0)
    # return it
    return img

def get_max_row(Fol_In, NX):
    """
    Function to calculate the maximum number of rows that a given PIV run can have.
    This is required to initialize a dummy for the smoothing.

    Parameters
    ----------
    Fol_In : string
        File path of the processed PIV data.
    NX : int
        Number of columns in the data.

    Returns
    -------
    NY : int
        Maximum number of rows in the data.

    """
    # create the input folder of the data
    Fol_In = os.path.join(Fol_In, 'data_files')
    # create a list of all the files
    files = os.listdir(Fol_In)
    # initialize a dummy to fill with the file sizes
    file_size = np.zeros((len(files)))
    # iterate over the list
    for i in range(0,file_size.shape[0]):
        # calculate the file size
        file_size[i] = os.stat(os.path.join(Fol_In, files[i])).st_size
    # get the maximum index
    idx_max = np.argmax(file_size)
    # get the name of the file with the largest size
    file_max = files[idx_max]
    # get the Number of Rows*Columns
    NXNY = np.genfromtxt(os.path.join(Fol_In, file_max))
    # calculate the number of columns
    NY = NXNY.shape[0]//NX
    # return the maximum number
    return NY

def filter_invalid(x, y, u, v, ratio, mask, valid_thresh):
    """
    Function to filter out invalid vectors near the top. This is done by calculating
    the percentage of valid vectors in a row, starting from the bottom. If a 
    row has less than the given threshold, it and everything above it gets sliced.

    Parameters
    ----------
    x : 2d np.array
        Array containg the x coordinates of the interrogation window centres
    y : 2d np.array
        Array containg the y coordinates of the interrogation window centres 
    u : 2d np.array
        Array containing the u displacement for every interrogation window
    v : 2d np.array
        Array containing the u displacement for every interrogation window
    ratio : 2d np.array
        Array containing the sig2noise ratio for every interrogation window.
    mask : 2d np.array
        Array containing the bit if the vector was repleaced for every
        interrogation window
    valid_thresh : int
        Threshold for the valid vectors (in percent) under which the row
        gets filtered out.

    Returns
    -------
    x : 2d np.array
        Array containg the x coordinates of the interrogation window centres
    y : 2d np.array
        Array containg the y coordinates of the interrogation window centres 
    u : 2d np.array
        Array containing the u displacement for every interrogation window
    v : 2d np.array
        Array containing the u displacement for every interrogation window
    ratio : 2d np.array
        Array containing the sig2noise ratio for every interrogation window.
    valid : 2d np.array, boolean
        Array containing 'True' if the vector was NOT repleaced for every
        interrogation window
    invalid : 2d np.array, boolean
        Array containing 'True' if the vector was repleaced for every
        interrogation window

    """
    # create the mask taken from the nans in the sig2noise ratio
    mask_nan = np.isfinite(ratio)
    # transform the bit array of the mask from the data into a boolean array
    mask = ~mask.astype('bool')
    # calculate the union of the two arrays, it is True if the vector as valid
    # and False otherwise
    valid = mask*mask_nan
    # iterate over all the rows
    for i in range(0,u.shape[0]):
        # help index because we are starting from the bottom
        index = u.shape[0]-i-1
        # check for validity
        if np.sum(valid[index,:])/valid.shape[1] < valid_thresh:
            # slice the arrays if invalid, the +2 is because of the help index
            v = v[index+2:]
            u = u[index+2:]
            x = x[index+2:]
            y = y[index+2:]
            ratio = ratio[index+2:]
            valid = valid[index+2:]
            # stop iterating
            break
    # calculate the array of the invalid vectors by negating the valid array
    invalid = ~valid.astype('bool')
    # return the result
    return x, y, u, v, ratio, valid, invalid

def high_pass(u, v, sigma, truncate):
    """
    Function to apply a high pass filter to the velocity field to visualize
    vorticies. For a more detailed description of truncate and sigma see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    
    It is important, that neither u nor v contain any nans, the nan filter
    should be run before that to ensure this.
    
    Parameters
    ----------
    u : 2d np.array
        Array containing the u displacement for every interrogation window
    v : 2d np.array
        Array containing the u displacement for every interrogation window
    sigma : float64
        Standard deviation for Gaussian kernel.
    truncate : float64
        Truncate the filter at this many standard deviations.

    Returns
    -------
    u_filt : 2d np.array
        Highpass filtered array containing the u displacement for every
        interrogation window
    v_filt : 2d np.array
        Highpass filtered array containing the u displacement for every
        interrogation window

    """
    # get the blurred velocity field
    u_blur = gaussian_filter(u, sigma = sigma, mode = 'nearest', truncate = truncate)
    v_blur = gaussian_filter(v, sigma = sigma, mode = 'nearest', truncate = truncate)
    # subtract to get the high pass filtered velocity field
    u_filt = u - u_blur
    v_filt = v - v_blur
    # return the result
    return u_filt, v_filt

def fill_zeros(u, v, NY_max):
    missing_rows = NY_max - u.shape[0]
    dummy_zeros = np.zeros((missing_rows, u.shape[1]))
    u = np.vstack((dummy_zeros, u))
    v = np.vstack((dummy_zeros, v))
    return u, v


# Fol_In = 'C:\PIV_Processed\Images_Processed\Rise_64_16_peak2RMS\Results_R_h1_f1200_1_p15_64_16'
# NX = get_column_amount(Fol_In)
# NY_max = get_max_row(Fol_In, NX)
# # Fol_Raw = get_raw_folder(Fol_In)
# # img = load_raw_image(Fol_Raw, 50)

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


def calc_qfield(x, y, u, v):
    u_smo, dum, dum, dum = smoothn.smoothn(u, s = 5)
    v_smo, dum, dum, dum = smoothn.smoothn(v, s = 5)
    # calculate the derivatives
    ux = np.gradient(u_smo, x[0,:], axis = 1)
    uy = np.gradient(u_smo, y[:,0], axis = 0)
    vx = np.gradient(v_smo, x[0,:], axis = 1)
    vy = np.gradient(v_smo, y[:,0], axis = 0)
    qfield = -0.5*(ux**2+2*vx*uy+vy**2)
    return qfield, u_smo, v_smo


Fol_In = 'D:\PIV_Processed\Images_Processed\Results_F_h4_f1200_1_s_24_24'
Fol_Raw = 'D:\PIV_Processed\Images_Preprocessed\F_h4_f1200_1_s'
NX = get_column_amount(Fol_In)
Fol_Img = create_folder('Temp')
Frame0 = 900
N_T = 1
# idx = 9
# plt.plot(y[:,0], u[:,idx])
# fil = sci.firwin(y.shape[0]//20, 0.0000000005, window='hamming', fs = 2)

# u_filt =sci.filtfilt(b = fil, a = [1], x = u, axis = 0, padlen = 3, padtype = 'constant')
# v_filt =sci.filtfilt(b = fil, a = [1], x = v, axis = 0, padlen = 3, padtype = 'constant')
# plt.plot(y[:,0], u_filt[:,idx])
import imageio
IMAGES_CONT = []
Gifname = 'qfield.gif'
for i in range(0, N_T):
    Load_Idx = Frame0+ i*1
    img = cv2.imread(Fol_Raw + os.sep + 'F_h4_f1200_1_s.%06d.tif'%Load_Idx, 0)
    x, y, u, v, ratio, mask = load_txt(Fol_In, Load_Idx, NX)
    u, v = high_pass(u, v, 3, 3)
    x, y, u, v = pad(x, y, u, v, 273)
    qfield, u_smo, v_smo = calc_qfield(x, y, u, v)
    fig, ax = plt.subplots(figsize = (4,8))
    # ax.imshow(np.flip(img, axis = 0), cmap = plt.cm.gray)
    ax.quiver(u_smo, v_smo, scale =3, color = 'lime')
    # ax.invert_yaxis()
    fig.savefig('test.jpg',dpi = 400)
    fig, ax = plt.subplots()
    cs = plt.pcolormesh(x,y,qfield, vmin=-0.0001, vmax=0, cmap = plt.cm.viridis) # create the contourplot using pcolormesh
    ax.set_aspect('equal') # set the correct aspect ratio
    clb = fig.colorbar(cs, pad = 0.2) # get the colorbar
    # clb.set_ticks(np.linspace(-100, 0, 6)) # set the colorbarticks
    clb.ax.set_title('Q Field \n [1/s$^2$]', pad=15) # set the colorbar title
    ax.contourf(qfield)
    ax.set_aspect(1)
    ax.set_ylim(0,1271)
    fig.tight_layout(pad=0.5)
    Name_Out = Fol_Img+os.sep+'contour%06d.png'%Load_Idx
    # fig.savefig(Name_Out, dpi=65)
    # # plt.close(fig)
    # IMAGES_CONT.append(imageio.imread(Name_Out))
    # print('Image %d of %d'%((i+1), N_T))
# imageio.mimsave(Gifname, IMAGES_CONT, duration = 0.05)
# import shutil
# shutil.rmtree(Fol_Img)