# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:16:43 2020

@author: manue
@description: This is a test function to get the PIV interrogation for rectangular windows

WARNING: This is an area that is not fully developed, do not change this code
"""


import numpy as np
import numpy.lib.stride_tricks
import cv2
import os
from numpy.fft import rfft2, irfft2, fftshift
import matplotlib.pyplot as plt
from openpiv import process, validation, filters, pyprocess, tools, preprocess,scaling
# not modified
def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which
    each slice, (along the first axis) is an interrogation window.

    """
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (sz * shape[1] * (window_size - overlap),
               sz * (window_size - overlap), sz * shape[1], sz)
    shape = (int((shape[0] - window_size) / (window_size - overlap)) + 1, int(
        (shape[1] - window_size) / (window_size - overlap)) + 1, window_size, window_size)

    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)



# modified
def moving_window_array2(array, win_width, win_height, overlap_width, overlap_height):
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (sz * shape[1] * (win_height - overlap_height),
                sz * (win_width - overlap_width), sz * shape[1], sz)
    shape = (int((shape[0] - win_height) / (win_height - overlap_height)) + 1, int(
        (shape[1] - win_width) / (win_width - overlap_width)) + 1, win_height, win_width)

    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, win_height, win_width)

# modified
def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which
    each slice, (along the first axis) is an interrogation window.

    """
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (sz * shape[1] * (window_size - overlap),
               sz * (window_size - overlap), sz * shape[1], sz)
    shape = (int((shape[0] - window_size) / (window_size - overlap)) + 1, int(
        (shape[1] - window_size) / (window_size - overlap)) + 1, window_size, window_size)

    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)
# modified
def get_coordinates(image_size, win_width, win_height, overlap_width, overlap_height):
        """Compute the x, y coordinates of the centers of the interrogation windows.

        Parameters
        ----------
        image_size: two elements tuple
            a two dimensional tuple for the pixel size of the image
            first element is number of rows, second element is 
            the number of columns.

        window_size: int
            the size of the interrogation windows.

        overlap: int
            the number of pixel by which two adjacent interrogation
            windows overlap.


        Returns
        -------
        x : 2d np.ndarray
            a two dimensional array containing the x coordinates of the 
            interrogation window centers, in pixels.

        y : 2d np.ndarray
            a two dimensional array containing the y coordinates of the 
            interrogation window centers, in pixels.

        """

        # get shape of the resulting flow field
        '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        The get_field_shape function calculates how many interrogation windows
        fit in the image in each dimension output is a 
        tuple (amount of interrogation windows in y, amount of interrogation windows in x)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        The get coordinates function calculates the coordinates of the center of each 
        interrogation window using bases on the to field_shape returned by the
        get field_shape function, the window size and the overlap. It returns a meshgrid
        of the interrogation area centers.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''

        field_shape = get_field_shape(image_size, win_width, win_height, overlap_width, overlap_height)

        # compute grid coordinates of the interrogation window centers
        x = np.arange(field_shape[1])*(win_width-overlap_width) + (win_width)/2.0
        y = np.arange(field_shape[0])*(win_height-overlap_height) + (win_height)/2.0

        return np.meshgrid(x, y[::-1])
# not modified
def find_first_peak(corr):
    """
    Find row and column indices of the first correlation peak.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map

    Returns
    -------
    i : int
        the row index of the correlation peak

    j : int
        the column index of the correlation peak

    corr_max1 : int
        the value of the correlation peak

    """
    ind = corr.argmax()
    s = corr.shape[1]

    i = ind // s
    j = ind % s

    return i, j, corr.max()


# not modified
def find_subpixel_peak_position(corr, subpixel_method='gaussian'):
        """
        Find subpixel approximation of the correlation peak.

        This function returns a subpixels approximation of the correlation
        peak by using one of the several methods available. If requested,
        the function also returns the signal to noise ratio level evaluated
        from the correlation map.

        Parameters
        ----------
        corr : np.ndarray
            the correlation map.

        subpixel_method : string
             one of the following methods to estimate subpixel location of the peak:
             'centroid' [replaces default if correlation map is negative],
             'gaussian' [default if correlation map is positive],
             'parabolic'.

        Returns
        -------
        subp_peak_position : two elements tuple
            the fractional row and column indices for the sub-pixel
            approximation of the correlation peak.
        """

        # initialization
        default_peak_position = (
                np.floor(corr.shape[0] / 2.), np.floor(corr.shape[1] / 2.))
        '''this calculates the default peak position (peak of the autocorrelation).
        It is window_size/2. It needs to be subtracted to from the peak found to determin the displacment
        '''
        #default_peak_position = (0,0)

        # the peak locations
        peak1_i, peak1_j, dummy =find_first_peak(corr)
        '''
        The find_first_peak function returns the coordinates of the correlation peak
        and the value of the peak. Here only the coordinates are needed.
        '''

        try:
            # the peak and its neighbours: left, right, down, up
            c = corr[peak1_i,   peak1_j]
            cl = corr[peak1_i - 1, peak1_j]
            cr = corr[peak1_i + 1, peak1_j]
            cd = corr[peak1_i,   peak1_j - 1]
            cu = corr[peak1_i,   peak1_j + 1]

            # gaussian fit
            if np.any(np.array([c, cl, cr, cd, cu]) < 0) and subpixel_method == 'gaussian':
                subpixel_method = 'centroid'

            try:
                if subpixel_method == 'centroid':
                    subp_peak_position = (((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) / (cl + c + cr),
                                          ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) / (cd + c + cu))

                elif subpixel_method == 'gaussian':
                    subp_peak_position = (peak1_i + ((np.log(cl) - np.log(cr)) / (2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr))),
                                          peak1_j + ((np.log(cd) - np.log(cu)) / (2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu))))

                elif subpixel_method == 'parabolic':
                    subp_peak_position = (peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                                          peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu))

            except:
                subp_peak_position = default_peak_position

        except IndexError:
            subp_peak_position = default_peak_position

            '''This block is looking for the neighbouring pixels. The subpixelposition is calculated based one
            the correlation values. Different methods can be choosen.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            This function returns the displacement in u and v
            '''
        return subp_peak_position[0] - default_peak_position[0], subp_peak_position[1] - default_peak_position[1]
# modified
def get_field_shape(image_size, win_width, win_height, overlap_width, overlap_height):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size: int
        the size of the interrogation window.
    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field
    """

    return ((image_size[0] - win_height) // (win_height - overlap_height) + 1,
            (image_size[1] - win_width) // (win_width - overlap_width) + 1)

# # example case for a very simple array to get the windows correctly
# dim = 8
# AA = np.zeros((dim,1))
# for i in range(0, dim):
#     AA[i]=i
# b = np.copy(AA)
# c=np.zeros((dim,1))
# for i in range(0,dim-1):
#     for k in range(0,dim):
#         c[k]=AA[k-i-1]
#     b = np.hstack((b,c))
    
# test = moving_window_array(b,4,2)
# test2 = moving_window_array2(b,8,2,4,1)

# set up the input folder
Fol_In = 'C:'+os.sep+'Users'+os.sep+'manue'+os.sep+'Desktop'+os.sep+'tmp'+os.sep
# load two images as a test
file_a = Fol_In + 'R_h1_f1200_1_p15.000280.tif'
file_b = Fol_In + 'R_h1_f1200_1_p15.000281.tif'
frame_a = cv2.imread(file_a,0)
frame_b = cv2.imread(file_b,0)

# set a rectangular window with an aspect ratio of 2 and an overlap of 50%
win_height = 48
win_width = int(0.5*win_height)
overlap_height = int(0.5*win_height)
overlap_width = int(0.5*win_width)

# calculate the correlation windows and the correlation map
cor_win_1_new = moving_window_array2(frame_a, win_width, win_height, overlap_width, overlap_height)
cor_win_2_new = moving_window_array2(frame_b, win_width, win_height, overlap_width, overlap_height)
correlation = fftshift(irfft2(np.conj(rfft2(cor_win_1_new)) *rfft2(cor_win_2_new)).real, axes=(1, 2))
# plt.contourf(corr[0,:,:])

# this part is taken from the windef.py file and modified to suit our needs
disp_new = np.zeros((np.size(correlation, 0), 2))#create a dummy for the loop to fill
for i in range(0, np.size(correlation, 0)):
    ''' determine the displacment on subpixel level '''
    disp_new[i, :] = find_subpixel_peak_position(
        correlation[i, :, :], subpixel_method='gaussian')
'this loop is doing the displacment evaluation for each window '

# shapes = np.array(get_field_shape(frame_a.shape, win_width, overlap_width))
shapes = get_field_shape(frame_a.shape, win_width, win_height, overlap_width, overlap_height)
u = disp_new[:, 1].reshape(shapes)
v = -disp_new[:, 0].reshape(shapes)
'reshaping the interrogation window to vector field shape'

x, y = get_coordinates(frame_a.shape, win_width, win_height, overlap_width, overlap_height)


##############################################################################
# load the solution of the square piv to compare
old = np.genfromtxt('C:\\Users\manue\Desktop\\tmp_processed\Open_PIV_results_test\\field_A000.txt')
nxny = old.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
n_s = 2 * nxny
## 1. Reconstruct Mesh from file
X_S = old[:, 0]
Y_S = old[:, 1]
# Number of n_X/n_Y from forward differences
GRAD_Y = np.diff(Y_S)
# Depending on the reshaping performed, one of the two will start with
# non-zero gradient. The other will have zero gradient only on the change.
IND_X = np.where(GRAD_Y != 0)
DAT = IND_X[0]
n_y = DAT[0] + 1
# Reshaping the grid from the data
n_x = (nxny // (n_y))  # Carefull with integer and float!
x_old = (X_S.reshape((n_x, n_y)))
y_old = (Y_S.reshape((n_x, n_y)))  # This is now the mesh! 60x114.
# Reshape also the velocity components
V_X = old[:, 2]  # U component
V_Y = old[:, 3]  # V component
# Put both components as fields in the grid
u_old = (V_X.reshape((n_x, n_y)))
v_old = (V_Y.reshape((n_x, n_y)))

# plot the v component of the velocity for different heights to get a comparison
for idx in range(0, x.shape[0]):
    fig, ax = plt.subplots()
    plt.scatter(x[0,:],v[idx,:], c='b', label = 'new')
    plt.plot(x[0,:],v[idx,:], c='b')
    plt.scatter(x_old[0,:],v_old[(idx+1)*2],c='r', label = 'old')
    plt.plot(x_old[0,:],v_old[(idx+1)*2],c='r')
    plt.grid()
    plt.legend()







