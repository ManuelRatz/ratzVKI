# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:59:47 2020
@author: koval
@description: All functions for the edge detection.
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

def integrate_curvature(y):
    x = np.linspace(-2.5,2.5,1000)
    dydx = np.gradient(y, x)
    ddyddx = np.gradient(dydx, x)
    curvature = ddyddx / (1 + (dydx)**2)**1.5
    ret = np.trapz(curvature, x)
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
    Path_Matrix = 'C:\Anna' + os.sep + case + os.sep + fluid
    Matrix = np.genfromtxt(Path_Matrix + os.sep + case + '_Matrix_' + fluid + '.txt', dtype = str)
    for i in range(Matrix.shape[0]):
        if Matrix[i,0] == test_case:
            break
    if case == 'Rise':
        Fol_Data = 'C:\Anna' + os.sep + case + os.sep + fluid + os.sep + test_case[:-2] + os.sep + test_case[-1:]
        Pressure = int(test_case[1:5])
        H_Final = float(Matrix[i,1])
        Speed = None
    elif case == 'Fall':
        invert = test_case[::-1]
        cut = invert[2:]
        Speed = cut[::-1]
        Fol_Data = 'C:\Anna' + os.sep + case + os.sep + fluid + os.sep + Speed + os.sep + test_case[-1:]
        Pressure = None
        H_Final = None
    else:
        raise ValueError('Invalid Case, must be *Rise* or *Fall*')
    Run = test_case[-1:]
    Frame0 = int(Matrix[i,2])
    Crop = np.array([Matrix[i,3], Matrix[i,4]], dtype = int)
    return Fol_Data, Pressure, Run, H_Final, Frame0, Crop, Speed

def load_image(name, crop_index, idx, load_idx, pressure, run, speed, cv2denoise):
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
    # at the beginning we need to cheat a little with cv2 because the highpass
    # filtered image alone is not enough so we do a little denoising for the 
    # first 10 steps
    if cv2denoise == True:
        img = cv2.fastNlMeansDenoising(img.astype('uint8'),1,1,7,21)
    # convert the image to integer to avoid bound errors
    img = img.astype(np.int)
    # calculate the blurred image
    blur = gaussian_filter(img, sigma = 4, mode = 'nearest', truncate = 3)
    # subtract the blur yielding the interface
    img_hp = img - blur
    # subtract values below 3, this is still noise, so we kill it
    img_hp[img_hp<3] = 0
    # return the image
    return img_hp, img

def edge_detection_grad(crop_img, threshold_pos, wall_cut, threshold_outlier,\
                        kernel_threshold, threshold_int, do_mirror):
    """
    Function to get the the edges of the interface

    Parameters
    ----------
    crop_img : 2d np.array
        Image as array in grayscale.
    treshold_pos : float64
        Required threshold for the gradient to be taken as a valid value.
    wall_cut : int
        How many pixels to cut near the wall.
    threshold_outlier : float64
        Percentage value indicating the relative difference between y and the mean at which to filter.
    do_mirror : boolean
        If True, the right side of the image will be mirrored because the gradient is stronger there.

    Returns
    -------
    grad_img : 2d np.array
        Image as a binary map indicating the edge as 1.
    y_index : 1d np.array
        Y-coordinates of the edge.
    x_index : 1d np.array
        X-coordinates of the edge.

    """
    grad_img = np.zeros(crop_img.shape)
    y_index  = (np.asarray([])) 
    Profiles = crop_img[:,wall_cut:len(crop_img[1])-wall_cut] # analyse the region where no spots from the wall disturb detection
    # Profiles_s = savgol_filter(Profiles, 15, 2, axis =0)        # intensity profile smoothened
    Profiles_d = np.gradient(Profiles, axis = 0)              # calculate gradient of smoothend intensity along the vertical direction
    idx_maxima = np.zeros((Profiles.shape[1]),dtype = int)
    for i in range(0,Profiles_d.shape[1]):
        # if i == 130:
        #     print('Stop')
        if np.max(Profiles_d[:,i]) <= threshold_pos:
            idx_maxima[i] = int(np.argmax(Profiles_d[5:,i]))+5
        else:
            idx_maxima[i] = int(np.argmax(Profiles_d[5:,i] > threshold_pos))+5             # find positions of all the maxima in the gradient
    # idx_maxima = np.argmax(Profiles_d > 10, axis = 0)                # find positions of all the maxima in the gradient
    #mean_a = np.mean(Profiles_s, axis=0)
    for j in range(Profiles_d.shape[1]):
        # accept only the gradient peaks that are above the threeshold (it avoids false detections lines where the interface is not visible)  
        if Profiles_d[idx_maxima[j],j] > 1:
            grad_img[idx_maxima[j],j+wall_cut] = 1 # binarisation
        # else:
            # print('Below Threshold')
    if do_mirror == True:
        grad_img = mirror_right_side(grad_img)     
    y_index, x_index = np.where(grad_img==1) # get coordinates of the interface
    # sort both arrays to filter out outliers at the edges
    x_sort_index = x_index.argsort()
    y_index = y_index[x_sort_index[:]]
    x_index = x_index[x_sort_index[:]]

    # filter out outliers
    y_average  = np.median(y_index)
    for k in range(len(y_index)):
        kernel_size = 2 # amount of points to sample for median
        y_kernel=get_kernel(k, y_index,kernel_size)
        if np.abs(y_index[k]-np.median(y_kernel))/np.median(y_kernel) > kernel_threshold:
            grad_img[int(y_index[k]),x_index[k]] = 0
            # print('Filtered by kernel')
        if np.abs(y_index[k]-y_average)/np.median(y_index)>threshold_outlier:
            grad_img[int(y_index[k]),x_index[k]] = 0
            # print('Filtered by threshold')
    
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

def saveTxt(fol_data, h_mm, h_cl_l, h_cl_r, angle_gauss, angle_cosh, pressure,\
            fit_coordinates_gauss, fit_coordinates_exp, test_case):                
    Fol_txt = create_folder(os.path.join(fol_data,'data_files_test'))
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
    if(k > kernel_size and k < len(y_index)-kernel_size): # points in the middle
        return y_index[k-kernel_size:k+kernel_size+1]
    elif(k <= kernel_size): # points at the left edge
        return y_index[0:2*kernel_size+1] 
    elif(k >= len(y_index)-kernel_size): # points at the right edge
        return y_index[len(y_index)-2*kernel_size-1:len(y_index)]
    else: # error message to be sure
        print('Error')
        return

def fitting_cosh(x_data, y_data, P0):
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
    # define the help function
    def func(x_func,a,b,c,d):
        return (np.cosh(np.abs(x_func)**a/b)-1)*c+d
    # calculate the values of the fit
    popt_cons, _ = curve_fit(func, x_data, y_data, p0 = P0, maxfev=25000, bounds\
                             = ([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf]))
    # calculate the fitted data
    y_fit = func(x_data, popt_cons[0], popt_cons[1], popt_cons[2], popt_cons[3])
    # return it
    return popt_cons

def fitting_cosh2(x_data, y_data):
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
    if y_data[0] < y_data[y_data.shape[0]//2]:
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

def fitting_cosh3(x_data, y_data):
    # define the help function
    def func(x_func,a,b):
        return (np.cosh(np.abs(x_func)**a/b)-1)
    # calculate the values of the fit
    popt_cons, _ = curve_fit(func, x_data, y_data, maxfev=25000, bounds\
                             = ([0.2,0.1],[10,20]))
    # return it
    return popt_cons

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.abs(np.diag(cov)))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.5)
    # plt.plot(X, mu, label='GPR')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')

def fitting_advanced(grad_img,pix2mm,l,sigma_f,sigma_y):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
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
    mu_s = mu_s[:,0] 
    
    return mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm

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
    Fps = 500
    abbrev = test_case[:-2]
    def read_lvm(path):
    #path_cam = folder + 'test-THS-cam_001.lvm'
        header    = 12  # number of header rows
        value     = 15  # gives the number of the loop
        with open(path) as alldata:
            line = alldata.readlines()[14]
        n_samples = int(line.strip().split('\t')[1])
    
        time = []
        voltages = []
        for i in range(value):
            with open(path) as alldata:                       #read the data points with the context manager
                lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
            t_temp        = [float(line.strip().split('\t')[0]) for line in lines] 
            v_temp        = [float(line.strip().split('\t')[1]) for line in lines]
            time          = np.append(time,t_temp)
            voltages      = np.append(voltages,v_temp)
        
        return time, voltages
    # paths
    path_cam = os.path.join(fol_data, test_case+'-cam_001.lvm')
    t_cam, v_cam = read_lvm(path_cam)
    path_pr = os.path.join(fol_data, test_case+'-pressure_001.lvm')
    [time, voltages] = read_lvm(path_pr)
    p_pressure = [voltage*208.73543056621196-11.817265775905382 for voltage in voltages]
    path_valve = os.path.join(fol_data, test_case+'-valve_001.lvm')
    t_valve, v_valve = read_lvm(path_valve)
    
    idx_start_cam = np.argwhere(v_cam > 3)[0]
    idx_start_valve = np.argwhere(v_valve > 4.5)[0]
    pressure_signal = p_pressure[int(idx_start_valve):]
    frame0 = int((t_valve[idx_start_valve]-t_cam[idx_start_cam])*Fps)
    return pressure_signal, frame0

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

