# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:59:47 2020
@author: koval & Manuel Ratz
@description: All functions for the edge detection (Anna). Additions made by 
    Manuel. Now includes the advanced fitting tool and the hyperbolic cosine
    for the fitting.
"""
###############################################################################
# Functions
###############################################################################

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os
import cv2
from scipy.ndimage import gaussian_filter
from numpy.linalg import inv
import matplotlib.pyplot as plt

def load_image(name, crop_index, idx, load_idx, pressure, run, speed, cv2denoise, fluid):
    """
    Function to load the image and highpass filter them for edge detection.

    Parameters
    ----------
    name : str
        Data path and prefix of the current run.
    crop_index : tuple of int
        4 coordinates to crop the images to.
    idx : int
        Current step counting from the beginning index.
    load_idx : int
        Current step counting from 0.

    Returns
    -------
    img_hp : 2d np.array of int
        Highpass filtered image.
    img : 2d np.array of int
        Raw Image.

    """
    # check if the test case is a rise or a fall
    if pressure is not None:
        Image_name = name + os.sep + 'images' + os.sep + str(pressure) + '_' + run + '%05d' % load_idx + '.png'
    elif speed is not None:
        Image_name = name + os.sep + 'images' + os.sep + speed + '_' + run + '%04d' % load_idx + '.png'
    else:
        raise ValueError('No pressure or speed defined, can not identify case')
    
    # load the image
    img=cv2.imread(Image_name,0)
    # crop the image
    img = img[:,crop_index[0]:crop_index[1]]
    # denoise the images slightly if desired
    if cv2denoise == True:
        img_den = cv2.fastNlMeansDenoising(img.astype('uint8'),1,1,7,21)
    # convert the image to integer to avoid bound errors
    img_den = img_den.astype(np.int)
    # this if else chain is adapted to include blur effects at the
    # top and bottom of the images. Depending on the image quality
    # these parameters have to be adjusted
    if pressure is not None and idx <= 50:
        if fluid == 'Water':
            sigma = 8
        elif fluid == 'HFE':
            sigma = 3
        else:
            raise ValueError('Wrong case, must be *Water* or *HFE*')
    else:
        sigma = 3
    if speed is not None:
        if fluid == 'Water':
            sigma = 2
        elif fluid == 'HFE':
            sigma = 2.5
        else:
            raise ValueError('Wrong case, must be *Water* or *HFE*')
    # calculate the blur
    blur = gaussian_filter(img_den, sigma = sigma, mode = 'nearest', truncate = 3)
    # subtract the blur yielding the interface
    img_hp = img_den - blur
    # subtract values below 4, this is still noise, so we kill it
    img_hp[img_hp<4] = 0
    # return the images
    return img_hp, img

def edge_detection_grad(crop_img, threshold_grad, wall_cut, threshold_outlier_in,
                        threshold_outlier_out, kernel_threshold_out, kernel_threshold_in,
                        do_mirror, fluid, idx):
    # create an image dummy to fill with the detected interface
    grad_img = np.zeros(crop_img.shape)
    # crop the image according to the wallcut
    Profiles = crop_img[:,wall_cut:len(crop_img[1])-wall_cut] 
    # calculate the gradient columnwise
    Profiles_d = np.gradient(Profiles, axis = 0) 
    # create a dummy to fill the interface positions 
    idx_maxima = np.zeros((Profiles.shape[1]),dtype = int)
    
    """
    This if else chain again is adjusted to respect the blur that occurs for HFE,
    the principle remains the same in both cases:
    We iterate over the width of the image and find the maximum gradient
    if it is larger than the specified threshold, we take the first value
    larger than the threshold to be the interface position. If not, the 
    interface position is taken as the maximum gradient. The +5 is required to
    avoid detecting the gradient at the top of the images
    """
    if fluid == 'HFE':
        for i in range(0, Profiles_d.shape[1]):
            if idx > 40 and idx < 160:
                idx_maxima[i] = int(np.argmax(Profiles[5:,i] > 0.5*np.max(Profiles[:,i])))+5
            else:
                if np.max(Profiles_d[:,i]) <= threshold_grad:
                    idx_maxima[i] = int(np.argmax(Profiles_d[5:,i]))+5
                else:
                    idx_maxima[i] = int(np.argmax(Profiles_d[5:,i] > threshold_grad ))+5
    elif fluid == 'Water':
        for i in range(0,Profiles_d.shape[1]):
            if np.max(Profiles_d[:,i]) <= threshold_grad:
                idx_maxima[i] = int(np.argmax(Profiles_d[5:,i]) )+5
            else:
                idx_maxima[i] = int(np.argmax(Profiles_d[5:,i] ))+5
    else:
        raise ValueError('Wrong fluid, must be *Water* or *HFE*')
        
    # fill the interface positions into the image dummy
    for j in range(Profiles_d.shape[1]): 
        grad_img[idx_maxima[j],j+wall_cut] = 1 
    # mirror the right signal if required, this is because of the shadow near
    # the left wall
    if do_mirror == True:
        grad_img = mirror_right_side(grad_img)     
    # get the interface coordinates
    y_index, x_index = np.where(grad_img==1)
    # sort both arrays to filter out outliers at the edges
    x_sort_index = x_index.argsort()
    y_index = y_index[x_sort_index[:]].astype(np.float64)
    x_index = x_index[x_sort_index[:]]

    
    """
    Here we filter the outliers. This is done with the 2 different types of 
    thresholds. One uses the global median, the other a local median based on
    a set of points defined by a kernel size. For the inner and outer points of 
    the image, different thresholds exist. This is to account for the different
    inclination that is present at these points. The points are replaced by nans
    if they are filtered
    """
    y_average  = np.median(y_index)
    for j in range(0, y_index.shape[0]):
        k = y_index.shape[0]-j-1
        if k > 0.1*y_index.shape[0] and k < 0.9*y_index.shape[0]:
            kernel_size = 5 # amount of points to sample for median
            y_kernel = get_kernel(k, y_index,kernel_size)
            if (np.abs(y_index[k]-np.nanmedian(y_kernel))) > kernel_threshold_in:
                grad_img[int(y_index[k]),x_index[k]] = 0
                y_index[k] = np.nan
                continue # skip the next step in case an outlier was found
            if np.abs(y_index[k]-y_average) > threshold_outlier_in:
                grad_img[int(y_index[k]),x_index[k]] = 0
                y_index[k] = np.nan
        else:
            kernel_size = 2 # amount of points to sample for median
            y_kernel = get_kernel(k, y_index,kernel_size)
            if (np.abs(y_index[k]-np.nanmedian(y_kernel))) > kernel_threshold_out:
                grad_img[int(y_index[k]),x_index[k]] = 0
                y_index[k] = np.nan
                continue # skip the next step in case an outlier was found
            if np.abs(y_index[k]-y_average) > threshold_outlier_out:
                grad_img[int(y_index[k]),x_index[k]] = 0
                y_index[k] = np.nan  
    # return the interface position, once with the image dummy as well as the
    # sorted x and y coordinates
    return grad_img, y_index, x_index

def mirror_right_side(array):
    """
    Function to mirror the right side of the image

    Parameters
    ----------
    array : 2d np.array
        Image to be mirrored.

    Returns
    -------
    mirrored : 2d np.array
        Mirrored image.

    """
    # take the right side of the array
    right = array[:,array.shape[1]//2+array.shape[1]%2:]
    # check if the width in pixels is even, if yes return the mirror
    if (array.shape[1]%2 == 0):
        mirrored = np.hstack((np.fliplr(right), right))
        # return the mirrored array
        return mirrored
    # if not we have to take the middle column
    middle = np.expand_dims(array[:,array.shape[1]//2], 1)
    # and add the right side to it, once normal and once flipped
    mirrored = np.hstack((np.fliplr(right), middle,right))
    # return the mirrored array
    return mirrored

def integrate_curvature(y):
    """
    Function to average the curvature over the interface
    Important Note: This is not divided by the channel width, this is taken 
    care of in the setup of the solution
    
    Parameters
    ----------
    y : 1d np.array
        Array containing the vertical coordinates of the interface in mm.

    Returns
    -------
    ret : float64
        Integrated curvature over the channel width.

    """
    # create the x points
    x = np.linspace(-2.5,2.5,1000)
    # calculate the first and second derivative
    dydx = np.gradient(y, x)
    ddyddx = np.gradient(dydx, x)
    # calculate the curvature
    curvature = ddyddx / (1 + (dydx)**2)**1.5
    # integrate the curvature to average
    ret = np.trapz(curvature, x)
    # return this
    return ret

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

def get_parameters(test_case, case, fluid):
    """
    Function to load the properties of one test case

    Parameters
    ----------
    test_case : str
        Name of the Case. For Water: P1500_C30_A, for HFE: P2000_A for example
    case : str
        'Rise' or 'Fall'.
    fluid : str
        'Water' or 'HFE'.

    Returns
    -------
    Fol_Data : str
        Path to the data files of the test case.
    Pressure : int
        Initial pressure of the facility.
    Run : str
        'A', 'B' or 'C'.
    H_Final : int
        Equilibrium height in mm.
    Frame0 : int
        First frame with a valid interface to detect.
    Crop : tuple of int
        Crop coordinates for the image in px.
    Speed : str
        Release Speed: 'fast', 'middle' or 'slow'.

    """
    # locate the file containing the test matrix and load it
    Path_Matrix = 'C:\Anna' + os.sep + case + os.sep + fluid
    Matrix = np.genfromtxt(Path_Matrix + os.sep + case + '_Matrix_' + fluid + '.txt', dtype = str)
    # find the current test case
    for i in range(Matrix.shape[0]):
        if Matrix[i,0] == test_case:
            break
    # for a rise we load pressure and final height
    if case == 'Rise':
        Fol_Data = 'C:\Anna' + os.sep + case + os.sep + fluid + os.sep + test_case[:-2] + os.sep + test_case[-1:]
        Pressure = int(test_case[1:5])
        H_Final = float(Matrix[i,1])
        Speed = None
    # for a rise we load the release speed
    elif case == 'Fall':
        invert = test_case[::-1]
        cut = invert[2:]
        Speed = cut[::-1]
        Fol_Data = 'C:\Anna' + os.sep + case + os.sep + fluid + os.sep + Speed + os.sep + test_case[-1:]
        Pressure = None
        H_Final = None
    else:
        raise ValueError('Invalid Case, must be *Rise* or *Fall*')
    # extract the run name
    Run = test_case[-1:]
    # get frame0
    Frame0 = int(Matrix[i,2])
    # get the crop coordinates
    Crop = np.array([Matrix[i,3], Matrix[i,4]], dtype = int)
    # return the test case parameters
    return Fol_Data, Pressure, Run, H_Final, Frame0, Crop, Speed


def saveTxt_fall(fol_data, h_mm, h_cl_r, angle_gauss, angle_cosh, test_case):   
    # create the output folder             
    Fol_txt = create_folder(os.path.join(fol_data,'data_files'))
    # save the data into the arrays
    np.savetxt(os.path.join(Fol_txt, test_case + '_h_avg.txt'), h_mm, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_h_cl_r.txt'), h_cl_r, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_ca_gauss.txt'), angle_gauss, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_ca_cosh.txt'), angle_cosh, fmt='%.6f')

def saveTxt(fol_data, h_mm, h_cl_l, h_cl_r, angle_gauss, angle_cosh, pressure,\
            fit_coordinates_gauss, fit_coordinates_exp, test_case): 
    # create the output folder                       
    Fol_txt = create_folder(os.path.join(fol_data,'data_files'))
    # save the data into the arrays
    np.savetxt(os.path.join(Fol_txt, test_case + '_h_avg.txt'), h_mm, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_h_cl_r.txt'), h_cl_r, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_h_cl_l.txt'), h_cl_l, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_ca_gauss.txt'), angle_gauss, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_ca_cosh.txt'), angle_cosh, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_pressure.txt'), pressure, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_gauss_curvature.txt'), fit_coordinates_gauss, fmt='%.6f')
    np.savetxt(os.path.join(Fol_txt, test_case + '_cosh_curvature.txt'), fit_coordinates_exp, fmt='%.6f')

def get_kernel(k,y_index,kernel_size):
    """
    Function to get a defined kernel of an array.
    
    Parameters
    ----------
    k : int
        Index of the current iteration.
    y_index : 1d nupmy array
        Indices of the gradients.
    kernel_size : int
        Half the kernel length.

    Returns
    -------
    y_index : 1d numpy array
        Sliced index array of size 2*kernel_size+1.

    """
    # these are the points in the middle, no padding is required here
    if(k > kernel_size and k < len(y_index)-kernel_size):
        return y_index[k-kernel_size:k+kernel_size+1]
    # points at the left edge, these will all have the same kernel
    elif(k <= kernel_size):
        return y_index[0:2*kernel_size+1] 
    # points at the right edge, these will have the same kernel as well
    elif(k >= len(y_index)-kernel_size):
        return y_index[len(y_index)-2*kernel_size-1:len(y_index)]
    # error message in case something went wrong
    else:
        raise ValueError('Error, kernel could not be found')
        return

def fitting_cosh(x_data, y_data):
    """
    Function to fit the experimental data to a hyperbolic cosine. The data has
    to be already extracted from the image to only be x and y values

    Parameters
    ----------
    x_data : 1d np.array
        X values of the channel centered around 0, not normalized.
    y_data : 1d np.array
        Y values of the detected interface, mean subtracted.

    Returns
    -------
    y_fit : 1d np.array
        Fitted y values of the detected interface, mean subtracted.

    """
    # we sort the x and y values by ascending x order
    x_sort_index = x_data.argsort()
    y_index = y_data[x_sort_index[:]].astype(np.float64)
    x_index = x_data[x_sort_index[:]]
    # check to see if the profile is inverted. This is done by comparing a
    # point in the middle to a point on the outside. Depending on the result
    # the interface is flipped for the fitting or not
    if y_index[0] < y_index[y_data.shape[0]//2]:
        inverted = True
        shift = np.max(y_data)
        y_fit = -(y_data-shift)
    else:
        inverted = False
        shift = np.min(y_data)
        y_fit = y_data-shift
    # define the help function
    def func(x_func,a,b):
        return (np.cosh(np.abs(x_func)**a/b)-1)
    # calculate the values of the fit
    popt_cons, _ = curve_fit(func, x_data, y_fit, p0 = [1.2, 2.34],\
                             bounds = ([0.1, 0.1],[500, 500]), maxfev=1000)
    # return it
    return popt_cons, shift, inverted

def fitting_advanced(grad_img,pix2mm,l,sigma_f,sigma_y):
    """
    Function to use Gaussian Regression on the data points. Detailed documentation
    can be found here: http://krasserm.github.io/2018/03/19/gaussian-processes/
    
    """
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_x_index = i_x.argsort()
    i_x = i_x[i_x_index[:]]
    i_y = i_y[i_x_index[:]]
    # i_x = i_x-0.5
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = grad_img.shape[1]*pix2mm # width of the image in mm 
    
    X_t=i_x_mm; X_train=np.expand_dims(X_t, axis=1)
    Y_t=i_y_mm; Y_train=np.expand_dims(Y_t, axis=1)
    #y_c=data['y_c'] # This is the clean data.
    X=np.linspace(0, 5, 1000, endpoint=True)
    X = X-0.5*pix2mm
    X_test=np.expand_dims(X, axis=1)
    # Note that the vectors are augmented to be of sixe n x 1
    
    # This is the Gaussian Fit
    mu_s, cov_s = posterior_predictive(X_test, X_train, Y_train, img_width_mm,sigma_f,sigma_y)
    

    # fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
    # # plt.scatter(X_train,Y_train,c='white',
    # #             marker='o',edgecolor='black',
    # #             s=10,label='Data')
    # plot_gp(mu_s, cov_s, X_test)
    # from gaussian_processes_util import plot_gp
    # plot_gp(mu_s, cov_s, X_test)
    mu_s = mu_s[:,0] 
    return mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    """
    Function to plot the result of the Gaussian Process
    """
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.abs(np.diag(cov)))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.5)
    # plt.plot(X, mu, label='GPR')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')


def vol_average(y,x,img_width_mm):
    """
    Function to calculate average height of the meniscus
    :grad_img: binary image, 1 at the edge of the meniscus
    :x: x coordinate of data for polynomial fitting
    :f: coordinates of the fitted curve
    
    """
    if len(y)==0:
        h_mm = np.nan
    else:
        vol = np.trapz(y, x)  # integrating the fitted line in 2D (?)
        h_mm = vol/(img_width_mm) 
    return h_mm

def contact_angle(y, x, side, pix2mm):
    """
    Function to calculate the contact angle
    :x: x coordinate of data for polynomial fitting
    :f: coordinates of the fitted curve
    :side: 0 for left side, -1 for right side
    :angle: initialisation for array
    
    """
    x = x + 0.5*pix2mm
    if side == 0:
        grad = (np.gradient(y,x)[side])
        alfa = np.arctan(grad)
        alfa_deg = alfa*180/np.pi
        angle = 90+alfa_deg
    else:
        grad = (np.gradient(y,x)[side])
        alfa = np.arctan(grad)
        alfa_deg = alfa*180/np.pi
        angle = 90-alfa_deg
    return angle

def load_labview_files(fol_data, test_case):
    """
    Function to load the lvm files and cut synchronize them.
    
    Note: This file was created by Domenico and not Manuel, so the documentation
    is relatively sparse.

    Parameters
    ----------
    fol_data : str
        Location of the .lvm files.
    test_case : str
        Name of the current test case. Must be a rising test case

    Returns
    -------
    pressure_signal : 1d np.array
        Pressure signal of the facility during the acquisition.
    frame0_exp : int
        Beginning index of the measurement. This is however not the frame0 used
        for the images, as the interface comes into the FOV later.

    """
    # acquisition frequency
    Fps = 500
    def read_lvm(path):
        header    = 12  # number of header rows
        value     = 15  # gives the number of the loop
        # read the points
        with open(path) as alldata:
            line = alldata.readlines()[14]
        n_samples = int(line.strip().split('\t')[1])
    
        time = []
        voltages = []
        for i in range(value):
            #read the data points with the context manager
            with open(path) as alldata: 
                lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
            t_temp        = [float(line.strip().split('\t')[0]) for line in lines] 
            v_temp        = [float(line.strip().split('\t')[1]) for line in lines]
            time          = np.append(time,t_temp)
            voltages      = np.append(voltages,v_temp)
        
        return time, voltages
    # path to the camera voltage
    path_cam = os.path.join(fol_data, test_case+'-cam_001.lvm')
    # read time and voltage of the camera
    t_cam, v_cam = read_lvm(path_cam)
    # path to the facility voltage
    path_pr = os.path.join(fol_data, test_case+'-pressure_001.lvm')
    # read time and voltage of the facility pressure
    [time, voltages] = read_lvm(path_pr)
    # calculate the pressure from the voltage
    p_pressure = [voltage*208.73543056621196-11.817265775905382 for voltage in voltages]
    # path to the valve voltage
    path_valve = os.path.join(fol_data, test_case+'-valve_001.lvm')
    # read time and voltage from the valve
    t_valve, v_valve = read_lvm(path_valve)
    
    # find the start index of the valve and the camera
    idx_start_cam = np.argwhere(v_cam > 3)[0]
    idx_start_valve = np.argwhere(v_valve > 4.5)[0]
    # shift the pressure signal
    pressure_signal = p_pressure[int(idx_start_valve):]
    # calculate the first frame
    frame0_exp = int((t_valve[idx_start_valve]-t_cam[idx_start_cam])*Fps)
    # return pressure signal and frame0_exp
    return pressure_signal, frame0_exp

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.
        
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''  
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    
    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    return mu_s, cov_s