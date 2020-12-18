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
from smoothn import smoothn
from scipy.ndimage import gaussian_filter
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")

def get_frame0(Fol_In):
    """
    Function to get the index of the first used image

    Parameters
    ----------
    Fol_In : str
        Input folder where the .txt files are located.

    Returns
    -------
    frame0 : int
        First valid index.

    """
    # navigate to the .txt files
    Fol = os.path.join(Fol_In, 'data_files')
    # get a list of all the .txts
    data_list = os.listdir(Fol)
    # take the first one (this is frame 0)
    run0 = data_list[0]
    # crop the name to get the integer
    frame0 = int(run0[6:12])
    # return it
    return frame0

def get_NT(Fol_In):
    """
    Function to get the number of images to process

    Parameters
    ----------
    Fol_In : str
        Input folder where the .txt files are located.

    Returns
    -------
    N_T : int
        Total amount of processed image pairs.

    """
    # get the length of the folder containing the .txts
    N_T = len(os.listdir(os.path.join(Fol_In, 'data_files')))
    # return the number
    return N_T

def pad(x, y, u, v, width):
    """
    Function to pad the velocity profile with zeros at the wall

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
    width : TYPE
        DESCRIPTION.

    Returns
    -------
    x_p : 2d np.array
        Array containg the padded x coordinates of the interrogation window
        centres
    y_p : 2d np.array
        Array containg the y padded coordinates of the interrogation window
        centres 
    u_p : 2d np.array
        Array containing the padded u displacement for every interrogation
        window
    v_p : 2d np.array
        Array containing the padded u displacement for every interrogation
        window

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

def calc_flux(x, v, Scale):
    """
    Function to calculate the flux from the velocity field. Careful, that x and
    v are padded, meaning we have added v = 0 at the boundary conditions.
    The result is not converted to mm, we integrate with pixels

    Parameters
    ----------
    x : 1d np.array 
        Horizontal coordinates of the channel in pixels.
    v : 1d np.array
        Padded vertical velocity across the width of the channel in mm/s.

    Returns
    -------
    q : int
        Flux across the width of the channel in mm^2/s.

    """
    # calculate the flux of the padded fields
    q = np.trapz(v, x)/Scale
    # return the result
    return q

def shift_grid(x, y, padded = True):
    """
    Function to shift the grid by half the interrogation window height and 
    width, as well as adding on additional row and column. This is required for
    pcolormesh.

    Parameters
    ----------
    x : 2d np.array
        Array containg the x coordinates of the interrogation window centres
    y : 2d np.array
        Array containg the y coordinates of the interrogation window centres 
    padded : bool, optional
        Whether or not the x and y values are padded. The default is True.

    Returns
    -------
    x_pco : 2d np.array
        Array containg the x coordinates of the interrogation window edges
    y_pco : 2d np.array
        Array containg the y coordinates of the interrogation window edges 

    """
    if padded == True:
        x = x[:,1:-1]
        y = y[:,1:-1]
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
    x_pco = np.hstack((x, np.ones((x.shape[0],1))*(np.max(x)+2*half_width)))
    y_pco = np.vstack((np.ones((1,y.shape[1]))*(np.max(y)+2*half_height),y))
    # return the result
    return x_pco, y_pco
    
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
    nx : int
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

def load_s2n(Fol_In, idx):
    """
    Function to load the sig2noise column of the current index.

    Parameters
    ----------
    Fol_In : str
        Input folder where the .txt files are located.
    idx : int
        Index of the .txt file to be loaded.

    Returns
    -------
    s2n_ratio : 1d np.array
        Signal to noise ratio of the image pair PIV Processing as a column of 
        NX*NY containing the individual columns stacked on top of each other.

    """
    Fol_In = Fol_In + os.sep + 'data_files'
    # set the file name
    file_name = Fol_In + os.sep + 'field_%06d.txt' % idx
    # load the data
    data = np.genfromtxt(file_name)
    s2n_ratio = data[:, 4]
    return s2n_ratio  

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

def get_smo_folder(Fol_In):
    """
    Function to get the folder where the smoothed 3d tensors of the velocity
    profile are located.
    
    Parameters
    ----------
    Fol_In : str
        Input folder where the .txt files are located.

    Returns
    -------
    Fol_Smo : str
        Input folder where the .npy files are located.

    """
    # get the indices of the backslashs
    backslashs = [i for i, a in enumerate(Fol_In) if a == os.sep]
    # get the name of the results folder
    result_folder = Fol_In[backslashs[-1]+1:]
    # cut the suffix and prefix
    cut_name = cut_processed_name(result_folder)
    # create the output path
    Fol_Smo = os.path.join('C:\PIV_Processed\Fields_Smoothed', 'Smoothed_'+cut_name)
    # return the result
    return Fol_Smo

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

def fill_dummy(u, v, NY_max, Fill_dummy):
    """
    Function to fill an array with a dummy to be able to copy into the tensor

    Parameters
    ----------
    u : 2d np.array
        Array containing the u displacement for every interrogation window
    v : 2d np.array
        Array containing the u displacement for every interrogation window
    NY_max : int
        Vertical shape of the tensor, maximum amount of rows.
    Fill_dummy : int
        Dummy to fill the array with.

    Returns
    -------
    u : 2d np.array
        Array containing the u displacement for every interrogation window,
        extended by the dummy to match tensor height
    v : 2d np.array
        Array containing the u displacement for every interrogation window,
        extended by the dummy to match tensor height

    """
    # calculate the number of missing rows
    missing_rows = NY_max - u.shape[0]
    # create a dummy to stack on top
    dummy = np.ones((missing_rows, u.shape[1]))*Fill_dummy
    # stack on top the u and v array
    u = np.vstack((dummy, u))
    v = np.vstack((dummy, v))
    # return the filled arrays
    return u, v

def fill_nan(x, y, NY_max):
    """
    Function to fill an array with nans to be able to copy into the tensor

    Parameters
    ----------
    x : 2d np.array
        Array containg the x coordinates of the interrogation window centres
    y : 2d np.array
        Array containg the y coordinates of the interrogation window centres 
    NY_max : int
        Vertical shape of the tensor, maximum amount of rows.

    Returns
    -------
    x : 2d np.array
        Array containg the x coordinates of the interrogation window centres,
        extended by the dummy to match tensor height
    y : 2d np.array
        Array containg the y coordinates of the interrogation window centres,
        extended by the dummy to match tensor height 

    """
    # calculate the number of missing rows
    missing_rows = NY_max - x.shape[0]
    # create a dummy to stack on top
    dummy_zeros = np.ones((missing_rows, x.shape[1]))*np.nan
    # stack on top the x and y array
    x = np.vstack((dummy_zeros, x))
    y = np.vstack((dummy_zeros, y))
    # return the filled arrays
    return x, y

Fill_Dum = 1000
def load_and_smooth(Fol_Sol, order = 15, valid_thresh = 0.3):
    """
    Function to load and smooth an entire run

    Parameters
    ----------
    Fol_Sol : str
        File path to the processed fields.
    order : int, optional
        Amount of base sines for the transformation, currently inactive.
        The default is 15.
    valid_thresh : float, optional
        Maximum percentage of invalid vectors in a row before it gets removed.
        The default is 0.3.

    Returns
    -------
    tensor_x : 3d np.array
        3d Tensor containing the x coordinates of the interrogation window
        centres, extended by the dummy to match tensor height
    tensor_y : 3d np.array
        3d Tensor containing the y coordinates of the interrogation window
        centres, extended by the dummy to match tensor height
    profiles_u : 3d np.array
        3d Tensor containing the unsmoothed v displacement for every
        interrogation window, extended by the dummy to match tensor height
    profiles_v : 3d np.array
        3d Tensor containing the unsmoothed u displacement for every
        interrogation window, extended by the dummy to match tensor height
    smoothed_tot : 3d np.array
        3d Tensor containing one smoothed velocity component of the field of 
        every timestep..

    """
    def smooth_horizontal_sin(profiles, x_local, Width):
        """
        Function to smooth the horizontal profiles using a sin transformation.
        For that we are computing a temporal base Psi of sin(x*n*pi/Width), n = 1,2,3,...
        from which we calculate a projection matrix via Psi*Psi.T.
        Multiplying this new matrix with the horizontal profiles gives a smoothed 
        version that satisfies the boundary conditions automatically.
        
        This makes the 'smoothn_horizontal' function obsolete
    
        Parameters
        ----------
        profiles : 3d np.array
            3d Tensor containing one velocity component of the field of 
            every timestep. The invalid positions are filled with 1000s.
        x_local : 1d np.array
            1d array containing the x coordinates of the velocity field in pixels.
        Width : int
            Width of the channel in pixels.
    
        Returns
        -------
        smoothed_array : 3d np.array
            3d Tensor containing one smoothed velocity component of the field of 
            every timestep.
    
        """
        # define the n'th base function
        def basefunc(x, n, width, norm):                                                                                    
            return np.sin(n*x*np.pi/width)/norm  
        # initialize Psi, we remove 5 degrees of freedom with this
        Psi = np.zeros((x_local.shape[0],order))
        # calculate the norm
        norm = np.sqrt(x_local[-1]/2)
        # fill the Psi columns with the sin profiles
        for i in range(1, Psi.shape[1]+1):
            Psi[:,i-1] = basefunc(x_local, i, x_local[-1], norm)
        # calculate the projection
        projection = np.matmul(Psi,Psi.T)
        # create a dummy to fill
        smoothed_array = np.ones(profiles.shape)*Fill_Dum
        # iterate in time
        for i in range(0, profiles.shape[0]):
            # iterate along y axis
            for j in range(0, profiles.shape[1]):
                # get the profile along x
                prof_hor = profiles[i,j,:]
                # check that we do not have one containing 100s
                if np.mean(prof_hor) > 0.8*Fill_Dum:
                    continue
                else:
                    smoothed_array[i,j,:] = np.matmul(projection, prof_hor)*8
                    # the multiplication by 8 is required for some reason to get
                    # the correct magnitude, at the moment the reason for this is
                    # unclear. For a Fall this has to be 12 for some reason.
        return smoothed_array
    # extract the constants from the images
    # give the first index and the number of files
    Frame0 = get_frame0(Fol_Sol)
    N_T = get_NT(Fol_Sol)
    NX = get_column_amount(Fol_Sol) # number of columns
    Fol_Raw = get_raw_folder(Fol_Sol)
    NY_max = get_max_row(Fol_Sol, NX) # maximum number of rows
    Height, Width = get_img_shape(Fol_Raw) # image width in pixels
    Scale = Width/5 # scaling factor in px/mm
    Dt = 1/get_frequency(Fol_Raw) # time between images
    Factor = 1/(Scale*Dt) # conversion factor to go from px/frame to mm/s
    # this is a fill dummy, meaning all invalid positions are filled with 1000
    Fill_Dum = 1000
    print('Loading Velocity Fields')
    """
    Load the velocity profiles and put them into the 3d tensor in which all profiles are stored.
    
    For that we load the velocity field for each time step and add 0 padding at the edges.
    Afterwards we fill the velocity field to the top with 100s because not every index
    has the full region of interest, so in order to be able to copy it into the 3d tensor
    we have to add dummys at the top. In case something messes up, we use 100 and not 
    np.nan because firwin cannot deal with nans
    """
    # N_T = 500
    # initialize the velocity profile tensors, the width is +2 because we have 0 padding
    profiles_u = np.zeros((N_T, NY_max, NX+2))
    profiles_v = np.zeros((N_T, NY_max, NX+2))
    tensor_x = np.zeros((N_T, NY_max, NX+2))
    tensor_y = np.zeros((N_T, NY_max, NX+2))
    
    for i in range(0, N_T):
        Load_Index = Frame0 + i # load index of the current file
        x, y, u, v, ratio, mask = load_txt(Fol_Sol, Load_Index, NX) # load the data from the txt
        x, y, u, v, ratio, valid, invalid = filter_invalid(x, y, u, v, ratio, mask, valid_thresh = 0.3) # filter invalid rows
        x, y, u, v = pad(x, y, u, v, Width) # 0 pad at the edges
        u = u*Factor
        v = v*Factor
        u, v = fill_dummy(u, v, NY_max, Fill_Dum) # fill the 2d velocity array with 1000s
        x, y = fill_nan(x, y, NY_max) # fill the 2d velocity array with 1000s
        profiles_v[i,:,:] = v # store in the tensor
        profiles_u[i,:,:] = u # store in the tensor
        tensor_x[i,:,:] = x # store in the tensor
        tensor_y[i,:,:] = y # store in the tensor
        
    """
    We now smooth along all three axiies, first timewise (axis 0) in the frequency
    domain, then along the vertical direction (axis 1) with robust smoothn and finally
    along the horizontal axis, again with smoothn.
    """
    
    print('Smoothing along the time axis')
    smoothed_freq = smooth_frequency(profiles_v, N_T)
    
    print('Smoothing along the vertical axis')
    smoothed_vert_freq = smooth_vertical(smoothed_freq)
    
    print('Smoothing along the horizontal axis with smoothn')
    smoothed_tot = smoothn_horizontal(smoothed_vert_freq)
    # filter the values filled with the dummy
    inv = np.isfinite(tensor_x)
    smoothed_tot[~inv] = np.nan    
    profiles_u[~inv] = np.nan
    profiles_v[~inv] = np.nan
    
    # print('Smoothing along the horizontal axis with cosine transformation')
    # smoothed_tot_sin = smooth_horizontal_sin(smoothed_vert_freq, x[0,:], Width)
    # # filter the values filled with the dummy
    # inv = smoothed_tot_sin > 0.8*Fill_Dum
    # smoothed_tot_sin[inv] = np.nan

    return tensor_x, tensor_y, profiles_u, profiles_v, smoothed_tot

def smooth_frequency(profiles, N_T):
    """
    Function to smooth the velocity profiles along the time axis in the frequency
    domain. For this we are using filtfilt and firwin from scipy. The principle
    as follows:
        1) For each row and column in the velocity field we extract the row in
           time (henceforth called timerow).
        2) Because not all timerows have data in them (some have 100s) we find
           the first index not equal to 100, this is our starting index. We then
           search for the second index not equal to 100. If there is none, all
           other points are valid, so we take all of those points after the first
        3) If there is a second one we check the number of steps in between.
           If it is below 50, we search again after the second peak as the array
           is either too short (then nothing happens) or we get a longer section
           that is valid for our smoothing
        4) We then smooth the extracted profile in time using a Firwin length
           of 1/30th of the profile length. The smoothed array is then returned

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of every timestep.
        The axiies are as follows:
            0 - time
            1 - y (vertical)
            2 - x (horizontal)

    Returns
    -------
    smoothed_array : 3d np.array
        3d Tensor containing one smoothed velocity component of the field of 
        every timestep.

    """
    def smooth_profile(tmp_prof):
        """
        Function to smooth a 1d profile in time using firwin and filtfilt.
        IMPORTANT: The profile must not contain nans otherwise filtfilt will
        only amplify this problem

        Parameters
        ----------
        tmp_prof : 1d np.array
            Array containing one velocity component at one position in time.

        Returns
        -------
        prof_filtered : 1d np.array
            Array containing one smoothed velocity component at one position in time.

        """
        # select the windows
        filter_poly = sci.firwin(tmp_prof.shape[0]//10, 0.01 , window='hamming', fs = 100)
        # filter with filtfilt
        prof_filtered = sci.filtfilt(b = filter_poly, a = [1], x = tmp_prof, padlen = 10, padtype = 'odd')
        returno, DUMMY, DUMMY, DUMMY = smoothn(prof_filtered, s = 0.5, isrobust = True)
        # return the filtered profile
        return returno
    def extract_profile(time_column):
        """
        Function to extract the longest velocity profile in time from the
        current time step.

        Parameters
        ----------
        time_column : 1 np.array of length N_T (total number of images)
            DESCRIPTION.

        Returns
        -------
        tmp_prof : 1d np.array
            Temporary array containing a velocity profile in time at a fixed
            point in space.
        id_min : int
            Lower index at which the array is sliced.
        id_max : int
            Upper index at which the array is sliced.

        """
        # get the first index at which we dont have 100
        id1 = np.argmax(time_column < 0.8*Fill_Dum)
        # get the next index after id1 at which we have 100
        id2 = np.argmax(time_column[id1:] > 0.8*Fill_Dum)
        if id1 == time_column.shape[0]-1:
            return np.array([0]), 0, 0
        # check if the timeprofile is long enough
        if id2 == 0:
            return time_column, 0, time_column.shape[0]
        if id2 < 10:
            # attempt to find a second non 100 index somewhere else
            id1_second_attempt = id1 + id2 + np.argmax(time_column[id1+id2+1:] < 0.8*Fill_Dum) + 1
            id2_second_attempt = id1_second_attempt + np.argmax(time_column[id1_second_attempt:] > 0.8*Fill_Dum)
            # np.argmax returns 0 if nothing was found, so test this here
            if id2_second_attempt == 0:
                # if true we set the second index to the last of the time_column
                id2_second_attempt = time_column.shape[0]-1
            # set minimum and maximum index accordingly
            id_min = id1_second_attempt
            id_max = id2_second_attempt
        else:
            # set minimum and maximum index accordingly
            id_min = id1
            id_max = id1 + id2
        # slice the time profile
        tmp_prof = time_column[id_min:id_max]
        # return the sliced array and the two indices
        return tmp_prof, id_min, id_max
    
    # start by copying the array to avoid editing the original array
    smoothed_array = np.copy(profiles)
    # iterate across the y axis
    for j in range(0, smoothed_array.shape[1]):
        # iterate across the x axis
        for i in range(0, smoothed_array.shape[2]):
            # if j == 80:
                # if i == 38:
                # print(' ')
            # extract the longest profile in time
            tmp_prof, id1, id2 = extract_profile(smoothed_array[:,j,i])
            # check if it is long enough for filtfilt
            if tmp_prof.shape[0] < 100:
                continue
            # smooth the respective profile in time and copy into the dummy array
            smoothed_array[id1:id2,j,i] = smooth_profile(tmp_prof)
    # return the smoothed array
    return smoothed_array

def smooth_vertical(profiles):
    """
    Function to smooth the 3d tensor along the vertical axis

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.

    Returns
    -------
    work_array_smoothed : 3d np.array
        3d Tensor containing smoothed one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s. 

    """
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*Fill_Dum
    # iterate over each time step
    for j in range(0, smoothed_array.shape[0]):
        # find the first index at which we dont have 100
        valid_idx = np.argmax(np.abs(profiles[j,:,1]) < 0.8*Fill_Dum)
        # smooth along the vertical axis sliced after the valid idx
        smoothed_array[j,valid_idx:,:], Dum, Dum, Dum = smoothn(profiles[j,valid_idx:,:], s = 0.7, axis = 0, isrobust = True)
    return smoothed_array

def smoothn_horizontal(profiles):
    """
    Function to smooth the profiles along the horizontal axis

    Parameters
    ----------
    profiles : 3d np.array
        3d Tensor containing one velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.

    Returns
    -------
    smoothed_array : 3d np.array
        3d Tensor containing one smoothed velocity component of the field of 
        every timestep. The invalid positions are filled with 1000s.

    """
    # create a dummy to fill
    smoothed_array = np.ones(profiles.shape)*Fill_Dum
    # iterate in time
    for i in range(0, profiles.shape[0]):
    # for i in range(0, 200):
        if i == 110:
            print(' ')
        # get the profile along x
        prof_time = profiles[i,:,:]
        valid_row = int(np.argwhere(prof_time[:, 5] != 1000)[0])
        prof_valid = prof_time[valid_row:,:]
        # check that we do not have one containing 100s
        # flip the signal
        dum_flip = np.flip(prof_valid, axis = 1)
        # pad the profile by extending it twice and flipping these extensions
        padded_profile = np.hstack((-dum_flip[:,:-1], prof_valid, -dum_flip[:,1:]))
        # smooth the new profile
        smoothed_pad, DUMMY, DUMMY, DUMMY = smoothn(padded_profile, s = 0.125, isrobust = True, axis = 1)
        # copy the sliced, smoothed profile into the tensor
        smoothed_array[i,valid_row:,:] = smoothed_pad[:,prof_time.shape[1]-1:2*prof_time.shape[1]-1]
    return smoothed_array  

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
    """
    

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

    Returns
    -------
    qfield : TYPE
        DESCRIPTION.

    """
    u_copy = np.copy(u)
    v_copy = np.copy(v)
    u_smo, dum, dum, dum = smoothn(u_copy, s = 7.5)
    v_smo, dum, dum, dum = smoothn(v_copy, s = 7.5)
    # calculate the derivatives
    ux = np.gradient(u_smo, x[-1,:], axis = 1)
    uy = np.gradient(u_smo, y[:,0], axis = 0)
    vx = np.gradient(v_smo, x[-1,:], axis = 1)
    vy = np.gradient(v_smo, y[:,0], axis = 0)
    qfield = -0.5*(ux**2+2*vx*uy+vy**2)
    return qfield

def high_pass(u, v, sigma, truncate, padded = True):
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
    if padded == True:
        u = u[:,1:-1]
        v = v[:,1:-1]
    # get the blurred velocity field
    u_blur = gaussian_filter(u, sigma = sigma, mode = 'nearest', truncate = truncate)
    v_blur = gaussian_filter(v, sigma = sigma, mode = 'nearest', truncate = truncate)
    # subtract to get the high pass filtered velocity field
    u_filt = u - u_blur
    v_filt = v - v_blur
    # return the result
    return u_filt, v_filt

def flux_parameters(smoothed_profile, x, Scale, case):
    """
    Function to get the minimum and maximum flux for the mass flux

    Parameters
    ----------
    smoothed_profile : 3d np.array
        3d Tensor containing the smoothed vertical velocity component of the
        field of every timestep. The invalid positions are filled with nans.
    x : 2d np.array
        x coordinates of the interrogation window centers in px.
    Scale : float64
        Conversion factor to go from px -> mm.
    case : str
        Two cases, Rise or Fall, this influences the maximum, as it should be
        0 for a fall.

    Returns
    -------
    q_max : int
        Upper bound for the flux, set as maximum in the plots.
    q_min : int
        Lower bound for the flux, set as minimum in the plots
    q_ticks : 1d np.array
        The ticks for the y axis in the plots.

    """
    # get the flux for each valid column
    flux_tot = calc_flux(x, smoothed_profile, Scale)
    # check for the cases
    if case == 'Fall':
        maximum_flux = 0
    elif case == 'Rise':
        maximum_flux = np.nanmax(flux_tot) 
    else:
        raise ValueError('Invalid Case, must be Rise or Fall')
    minimum_flux = np.nanmin(flux_tot)
    if (maximum_flux-minimum_flux) > 1500:
        increments = 250        
    elif (maximum_flux-minimum_flux) > 1000:
        increments = 200
    else:
        increments = 100
    divider_max = int(np.ceil(maximum_flux/increments))
    flux_max = divider_max*increments
    divider_min = int(np.floor(minimum_flux/increments))
    flux_min = divider_min*increments
    flux_ticks = np.linspace(flux_min, flux_max, divider_max - divider_min+1)
    return  flux_max, flux_min, flux_ticks

def profile_parameters(smoothed_profile, case):
    # increments of the vertical axis
    increments = 50
    # calculate the maximum and minimum velocity of the profile
    minimum_v = np.nanmin(smoothed_profile)
    if case == 'Fall':
        maximum_v = 0
    elif case == 'Rise':
        maximum_v = np.nanmax(smoothed_profile)
    else:
        raise ValueError('Invalid Case, must be Rise or Fall')
    # get the integer, rounded up, that is the maximum when multiplied with increments
    divider_max = int(np.ceil(maximum_v/increments))
    v_max = divider_max*increments
    # get the integer, rounded down, that is the minimum when multiplied with increments
    divider_min = int(np.floor(minimum_v/increments))
    v_min = divider_min*increments
    # calculate the ticks
    v_ticks = np.linspace(v_min, v_max, divider_max - divider_min+1)      
    # calculate the expanded ticks, this is necessary, because the legend might
    # block the view of the profiles for some time steps
    v_ticks_expanded = np.copy(v_ticks)
    if (v_ticks[0] - v_min) > -50:
        v_ticks_expanded = np.hstack((np.array([v_ticks[0]-increments]), v_ticks))
    # return the result
    return v_max, v_min, v_ticks, v_ticks_expanded

def height_parameters(h_mm):
    # set the increments of the vertical axis
    increments = 5
    # calculate the maximum h
    maximum_h = np.max(h_mm)
    # calculate the maximum axis height as a multiple of 5 by rounding up
    divider_max = int(np.ceil(maximum_h/increments))
    h_max = divider_max*increments
    # calculate the ticks
    h_ticks = np.linspace(0, h_max, divider_max+1)
    # return the result, the minimum height is always 0, so it is not calculated
    return h_max, h_ticks
    
def first_valid_row(array):
    # get the indices of the nans
    nans = np.argwhere(np.isnan(array[:,5]))
    # return the length of this array, this will be the first index, that is not a nan
    return nans.shape[0]

def get_v_avg_max(profiles, x_tensor, Width, case):
    # check that a valid profile was given
    if (len(profiles.shape) != 3) or (len(x_tensor.shape) != 3):
        raise ValueError('Velocity and X values must be a 3d tensor')
    # check the case
    if case == 'Rise':
        # calculate the average velocity of the bottom 5 rows at timestep 0
        v_avg_max = np.trapz(profiles[0,-5:,:], x_tensor[0,-1,:], axis = -1)/Width
        # average the 5 velocities
        mean_vel_max = np.mean(v_avg_max)
    elif case == 'Fall':
        # calculate the average velocity of the second row from the bottom
        v_avg = np.trapz(profiles[:,-2,:], x_tensor[0,-1,:])/Width
        # apply heavy smoothing to the velocity, this is possible, because the 
        # profile is relatively smooth already, so we can smooth heavily
        v_avg_filter, DUMMY, DUMMY, DUMMY = smoothn(v_avg, s = 40, isrobust = True)
        # find the maximum
        mean_vel_max = np.nanmax(np.abs(v_avg_filter))
    else:
        raise ValueError('Invalid Case, must be Rise or Fall')
    # return the result
    return mean_vel_max
    
def get_acc_max(profiles, x_tensor, Dt, Width, case):
    """UNDER DEVELOPMENT, CAN STILL BE DONE LATER"""
    # check the case
    if case == 'Rise':
        # calculate the average velocity of the bottom 5 rows for the first 30 timesteps
        v_avg = np.trapz(profiles[:500,-5:,:], x_tensor[0,-1,:], axis = -1)/Width
        # average along the vertical axis to get the average velocity as a
        # function of time
        v_mean = np.mean(v_avg, axis = 1)
        # smooth this velocity heavily
        v_filter, DUMMY, DUMMY, DUMMY = smoothn(v_mean, s = 10, isrobust = True)
        # calculate the acceleration
        acc = np.gradient(v_filter, Dt)
        # the maximum acceleration is in the beginning
        acc_max = np.nanmax(np.abs(acc))
    elif case == 'Fall':
        v_avg = np.trapz(profiles[:,-2,:], x_tensor[0,-1,:])/Width
        v_avg_filter, DUMMY, DUMMY, DUMMY = smoothn(v_avg, s = 40, isrobust = True)
        acc = np.gradient(v_avg_filter, Dt)
        acc_max = np.nanmax(np.abs(acc))
    else:
        raise ValueError('Invalid Case, must be Rise or Fall')
    return acc_max

def save_run_params(path, run, vmax, t_end, frame0, nx, frequency, scale, height, width, accel):
    geo_format = '{:=7.4f}'
    header = """Run:                     """ + run + """
Duration:               """ + geo_format.format(t_end) + ' (s)' +"""
Frame 0:                 """ + str(frame0) + """
Number of columns:       """ + str(nx) + """
Acquisition Frequency:   """ + str(int(frequency)) + ' (Hz)' +"""
Scaling Factor:          """ + geo_format.format(scale) + ' (Px/mm)' +"""
Image Height:            """ + str(height) + ' (Px)' +"""
Image Width:             """ + str(width) + ' (Px)' + """
Max Avg. Velocity:       """ + geo_format.format(vmax) + ' (m/s)' +"""
Max Avg. Acceleration:   """ + geo_format.format(accel) + ' (m/s²)' 

    with open(path,"w+") as thefile:
        thefile.write(header) # write the header