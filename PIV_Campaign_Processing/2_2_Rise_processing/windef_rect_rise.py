# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:04:04 2019

@author: Manuel Ratz
@description: This is a modified version of the windef file. It
    adds to the functions of the windef file by using an adaptive
    ROI based on the mass flux. Another major difference is the
    possibility of rectangular windows. Run R_h1_f750_p15 should
    be skipped becauses the displacement is too large.
    
Summary of changes:
    - dynamic shifting ROI for the rising interface
    - saving of the current interface position
    - rectangular interrogation windows
    - normalization of the correlation map
    - sig2noise can be done peak2RMS
"""


import os
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
import numpy.lib.stride_tricks
import scipy.ndimage as scn
from scipy.interpolate import RectBivariateSpline
from openpiv import validation, filters, pyprocess, tools, preprocess,scaling
from openpiv import smoothn
import tools_patch_rise
import validation_patch
import matplotlib.pyplot as plt

def piv(settings):
    def func(args):
        """A function to process each image pair."""
        file_a, file_b, counter = args
        ' read images into numpy arrays'
        frame_a = tools_patch_rise.imread(os.path.join(settings.filepath_images, file_a))
        frame_b = tools_patch_rise.imread(os.path.join(settings.filepath_images, file_b))
        
        #%%
        """This is the part for the changing ROI, the new position of the interface
        is calculated after the final pass to avoid having to load the txt file.
        For the very first image pair we set the ROI to be the bottom 450 pixels.
        Afterwards the script will calculate the current position of the interface
        and then shift the ROI accordingly. If the Interface comes into view,
        the ROI gets shifted
        """

        if counter == settings.beginning_index:
            settings.ROI[0] = frame_a.shape[0]-settings.init_ROI-450
            settings.ROI[1] = frame_a.shape[0]
            settings.ROI[2] = 0
            settings.ROI[3] = frame_a.shape[1]
            settings.current_pos = settings.ROI[0]
        # if the interface comes into view again, set the ROI to the interface position
        if settings.current_pos > 0:
            settings.ROI[0] = int(settings.current_pos)
        # plot the current ROI on the images in case this is desired for animation purposes
        if settings.plot_ROI == True:
            plot_shifted_ROI(frame_b, counter, settings.current_pos, save_path)
        #%%
        """This is the part for the changing ROI, the new position of the interface
        is calculated after the final pass to avoid having to load the txt file.
        For the very first image pair we set the ROI to be the bottom 450 pixels.
        Afterwards the script will calculate the current position of the interface
        and then shift the ROI accordingly. If the Interface comes into view,
        the ROI gets shifted
        """
        settings.height_settings_call = np.array([settings.window_height[0], settings.overlap_height[0]])
        if (5*settings.height_settings_call[1] > (settings.ROI[1]-settings.ROI[0])):
            settings.height_settings_call = (settings.window_height[1], settings.overlap_height[1])
            if (5*settings.height_settings_call[1] > (settings.ROI[1]-settings.ROI[0])):
                return settings.current_pos, False
        ' crop to ROI'   
        frame_a =  frame_a[settings.ROI[0]:settings.ROI[1],settings.ROI[2]:settings.ROI[3]]
        frame_b =  frame_b[settings.ROI[0]:settings.ROI[1],settings.ROI[2]:settings.ROI[3]]
        if settings.dynamic_masking_method=='edge' or settings.dynamic_masking_method=='intensity':    
            frame_a = preprocess.dynamic_masking(frame_a,method=settings.dynamic_masking_method,filter_size=settings.dynamic_masking_filter_size,threshold=settings.dynamic_masking_threshold)
            frame_b = preprocess.dynamic_masking(frame_b,method=settings.dynamic_masking_method,filter_size=settings.dynamic_masking_filter_size,threshold=settings.dynamic_masking_threshold)

        '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
        'first pass'
        x, y, u, v, sig2noise_ratio = first_pass(frame_a,frame_b, settings.window_width[0], settings.height_settings_call[0],
                                                 settings.overlap_width[0], settings.height_settings_call[1], settings.iterations,
                                      correlation_method=settings.correlation_method, subpixel_method=settings.subpixel_method, do_sig2noise=settings.extract_sig2noise,
                                      sig2noise_method=settings.sig2noise_method, sig2noise_mask=settings.sig2noise_mask,)
    
        mask=np.full_like(x,False)
        if settings.validation_first_pass==True:    
            u, v, mask_g = validation.global_val( u, v, settings.MinMax_U_disp, settings.MinMax_V_disp)
            u,v, mask_s = validation.global_std( u, v, std_threshold = settings.std_threshold )
            u, v, mask_m = validation.local_median_val( u, v, u_threshold=settings.median_threshold, v_threshold=settings.median_threshold, size=settings.median_size )
            if settings.extract_sig2noise==True and settings.iterations==1 and settings.do_sig2noise_validation==True:
                u,v, mask_s2n = validation.sig2noise_val( u, v, sig2noise_ratio, threshold = settings.sig2noise_threshold)
                mask=mask+mask_g+mask_m+mask_s+mask_s2n
            else:
                mask=mask+mask_g+mask_m+mask_s
        'filter to replace the values that where marked by the validation'
        if settings.iterations>1:
             u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
             'adding masks to add the effect of all the validations'
             if settings.smoothn==True:
                  u,dummy_u1,dummy_u2,dummy_u3=smoothn.smoothn(u,s=settings.smoothn_p)
                  v,dummy_v1,dummy_v2,dummy_v3=smoothn.smoothn(v,s=settings.smoothn_p)        
        elif settings.iterations==1 and settings.replace_vectors==True:    
             u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
             'adding masks to add the effect of all the validations'
             if settings.smoothn==True:
                  u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
                  u,dummy_u1,dummy_u2,dummy_u3=smoothn.smoothn(u,s=settings.smoothn_p)
                  v,dummy_v1,dummy_v2,dummy_v3=smoothn.smoothn(v,s=settings.smoothn_p)        
     
        i = 1
        'all the following passes'
        for i in range(2, settings.iterations+1):
            x, y, u, v, sig2noise_ratio, mask = multipass_img_deform(frame_a, frame_b, settings.window_width[i-1], settings.window_height[i-1],
                                                    settings.overlap_width[i-1], settings.overlap_height[i-1],settings.iterations,i,
                                                    x, y, u, v, correlation_method=settings.correlation_method,
                                                    subpixel_method=settings.subpixel_method, do_sig2noise=settings.extract_sig2noise,
                                                    sig2noise_method=settings.sig2noise_method, sig2noise_mask=settings.sig2noise_mask,
                                                    MinMaxU=settings.MinMax_U_disp,
                                                    MinMaxV=settings.MinMax_V_disp,std_threshold=settings.std_threshold,
                                                    median_threshold=settings.median_threshold,median_size=settings.median_size,filter_method=settings.filter_method,
                                                    max_filter_iteration=settings.max_filter_iteration, filter_kernel_size=settings.filter_kernel_size,
                                                    interpolation_order=settings.interpolation_order)
            # If the smoothing is active, we do it at each pass
            if settings.smoothn==True:
                 u,dummy_u1,dummy_u2,dummy_u3= smoothn.smoothn(u,s=settings.smoothn_p)
                 v,dummy_v1,dummy_v2,dummy_v3= smoothn.smoothn(v,s=settings.smoothn_p)        
   
        
        '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
        if settings.extract_sig2noise==True and i==settings.iterations and settings.iterations!=1 and settings.do_sig2noise_validation==True:
            u,v, mask_s2n = validation_patch.sig2noise_val( u, v, sig2noise_ratio, threshold_low = settings.sig2noise_threshold)
            mask=mask+mask_s2n
        if settings.replace_vectors==True:
            u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
        #%%
        'Calculate the mean displacement, this is the variable that we use at the top'
        settings.current_pos = settings.current_pos - calc_disp(x, v, frame_b.shape[1])
        if settings.current_pos == np.nan:
            return settings.current_pos, False
        #%%
        'pixel/frame->pixel/sec'
        u=u/settings.dt
        v=v/settings.dt
        'scales the results pixel-> meter'
        x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = settings.scaling_factor )     
        'save to a file'
        save(x, y, u, v,sig2noise_ratio, mask ,os.path.join(save_path_txts,'field_%06d.txt' % counter), delimiter='\t')
        'some messages to check if it is still alive'
        # disable the grid in case it is activated
        plt.rcParams['axes.grid'] = False
        'some other stuff that one might want to use'
        if settings.show_plot==True or settings.save_plot==True:
            plt.close('all')
            plt.ioff()
            Name = os.path.join(save_path_images, 'Image_%06d.png' % counter)
            display_vector_field(os.path.join(save_path_txts, 'field_%06d.txt' % counter), scale=settings.scale_plot)
            if settings.save_plot==True:
                plt.savefig(Name, dpi=100)
            if settings.show_plot==True:
                plt.show()

        print('Image Pair ' + str(counter) + ' of ' + settings.save_folder_suffix)
        return settings.current_pos, True
    'Below is code to read files and create a folder to store the results'
    save_path=os.path.join(settings.save_path,'Results_'+settings.save_folder_suffix+'_'\
                           +str(settings.window_height[settings.iterations-1])+'_'+str(settings.window_width[settings.iterations-1]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_images = save_path + os.sep + 'velocity_fields'
    if not os.path.exists(save_path_images):
        os.makedirs(save_path_images)
    save_path_txts = save_path + os.sep + 'data_files'
    if not os.path.exists(save_path_txts):
        os.makedirs(save_path_txts)       
    # save the settings of the processing
    save_settings(settings, save_path)
    # set the initial ROI to be nothing, this is to avoid an error that results 
    # from the ROI being set to 'full'
    settings.ROI = np.array([0,0,0,0])
    
    task = tools_patch_rise.Multiprocesser(
        data_dir=settings.filepath_images, pattern_a=settings.frame_pattern_a, pattern_b=settings.frame_pattern_b)

    task.run(save_path, func=func, beginning_index = settings.beginning_index, n_cpus=1)

def plot_shifted_ROI(frame_b, counter, interface_position, save_path):
    """
    Function to plot frame b with the top of the ROI visible

    Parameters
    ----------
    frame_b : 2d np.array
        Array containing the second image.
    counter : int
        Index of the curent first image.
    interface_position : float64
        Current position of the interface in px. If the value is positive, the
        interface is visible.
    save_path : string
        Location where the results are stored.
    """
    
    # create the folder to store the ROI images into
    Fol_ROI = save_path + os.sep + 'ROI_Images' + os.sep
    if not os.path.exists(Fol_ROI):
        os.makedirs(Fol_ROI)
    # create the figure
    fig, ax = plt.subplots(figsize=(4,10))
    # show the image
    plt.imshow(frame_b, cmap=plt.cm.gray)
    # plot the interface line in case it is in the FOV
    if interface_position > 0:
        interface_line = np.ones((frame_b.shape[1],1))*interface_position
        ax.plot(interface_line, lw = 1)
    # save and close the figure
    fig.savefig(Fol_ROI + 'ROI_img_%06d.png' %counter, dpi = 75)
    plt.close(fig)
    return
    
def calc_disp(x, v, img_width):
    """
    Function to calculate the mean displacement of the current index

    Parameters
    ----------
    x : 2d np.array
        Array containing the x coordinates of the interrogation window centres.
    v : 2d np.array
        Array containing the v values of the interrogation windows.
    img_width : int
        Width of the images in px.

    Returns
    -------
    mean_disp : float64
        Mean displacement of the bottom 5 rows.

    """
    # create a vertical dummy of 0s for the padding
    dummy_0 = np.zeros((v.shape[0],1))
    # create a vertical dummy of img_width for the padding of x
    dummy_x_max = np.ones((x.shape[0],1))*img_width
    # pad the x values
    x_padded = np.hstack((dummy_0, x, dummy_x_max))
    # pad the v values
    v_padded = np.hstack((dummy_0, v, dummy_0))
    # calculate the mean displacement with np.trapz
    
    mean_disp = np.mean(np.trapz(v_padded[-5:,:], x_padded[-5:,:]))/img_width
    # return the mean_disp
    return mean_disp

def save_settings(settings, save_path):
    """
    Function to save the settings given in the client.

    Parameters
    ----------
    settings : class
        Class containing all the settings given.
    save_path : string
        Path where to save the generated txt file.
    """
    # extract the variables
    variables = vars(settings)
    # open the file
    with open(save_path+os.sep+"settings.txt", "w") as f:
        # iterate over the list of variables
        for key, value in variables.items():
            #write into the file
            f.write('{}'.format(key) + "\t" + '{}'.format(value)+"\n")

def correlation_func(cor_win_1, cor_win_2, win_width, win_height ,correlation_method='circular'):
    '''This function is doing the cross-correlation. Right now circular cross-correlation
    That means no zero-padding is done
    the .real is to cut off possible imaginary parts that remains due to finite numerical accuarcy
    
    Manuel: The function is modified to include the different heights and widths
        for the linear cross correlation, the circular one is not affected by this.
        Also now the mean is subtracted for both circular and linear and the correlation
        map is normalised
     '''
    cor_win_1 = cor_win_1-cor_win_1.mean(axis=(1,2)).reshape(cor_win_1.shape[0],1,1)
    cor_win_2 = cor_win_2-cor_win_2.mean(axis=(1,2)).reshape(cor_win_1.shape[0],1,1)
    
    if correlation_method=='linear':
        corr = fftshift(irfft2(np.conj(rfft2(cor_win_1,s=(2*win_height,2*win_width))) *
                                  rfft2(cor_win_2,s=(2*win_height,2*win_width))).real, axes=(1, 2))
        corr=corr[:,win_height//2:3*win_height//2,win_width//2:3*win_width//2]
    else:
        corr = fftshift(irfft2(np.conj(rfft2(cor_win_1)) *
                                  rfft2(cor_win_2)).real, axes=(1, 2))
    
    """
    Normalize the whole thing. For that we calculate the vector of length equal
    to the amount of correlation windows, that contains the following multiplication:
        
    simga_A*sigma_B*Window_Height*Window_Width
    
    This will give us a maximum of 1 for the correlation peak.
    """
    normalizer = (np.std(cor_win_2, axis=(1,2))*np.std(cor_win_1, axis=(1,2))\
                  *win_width*win_height)
    # normalize by expanding the dimensions of the normalizer, to be of shape (n_windows, 1, 1)
    corr_norm = corr / np.expand_dims(normalizer, axis = (1,2))
    # eliminate the values below 0 to get final correlation map and return it
    corr_norm[corr_norm<0]=0    
    return corr_norm

def frame_interpolation(frame, x, y, u, v, interpolation_order=1):
    '''This one is doing the image deformation also known as window deformation
    Therefore, the pixel values of the old image are interpolated on a new grid that is defined
    by the grid of the previous pass and the displacment evaluated by the previous pass
    '''
    '''
    The interpolation function dont like meshgrids as input. Hence, the the edges
    must be extracted to provide the sufficient input, also the y coordinates need
    to be inverted since the image origin is in the upper left corner and the
    y-axis goes downwards. The x-axis goes to the right.
    '''
    frame=frame.astype(np.float32)
    y1 = y[:, 0] # extract first coloumn from meshgrid
    y1 = y1[::-1] #flip 
    x1 = x[0, :] #extract first row from meshgrid
    side_x = np.arange(0, np.size(frame[0, :]), 1) #extract the image grid
    side_y = np.arange(0, np.size(frame[:, 0]), 1)

    ip = RectBivariateSpline(y1, x1, u) #interpolate the diplacement on the image grid
    ut = ip(side_y, side_x)# the way how to use the interpolation functions differs
                            #from matlab 
    ip2 = RectBivariateSpline(y1, x1, v)
    vt = ip2(side_y, side_x)
    
    '''This lines are interpolating the displacement from the interrogation window
    grid onto the image grid. The result is displacment meshgrid with the size of the image.
    '''
    x, y = np.meshgrid(side_x, side_y)#create a meshgrid 
    frame_def = scn.map_coordinates(
        frame, ((y+vt, x+ut,)), order=interpolation_order,mode='nearest')
    #deform the image by using the map coordinates function
    '''This spline interpolation is doing the image deformation. This one likes meshgrids
    new grid is defined by the old grid + the displacement.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    This function returns the deformed image.
    '''
    #print('stop')
    return frame_def


def moving_window_array_rectangular(array, win_width, win_height, overlap_width, overlap_height):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which
    each slice, (along the first axis) is an interrogation window.
    
    Manuel: The function is taken from pyprocess.py and modified to include different
        height and width and also their respective overlap
    """
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)
    strides = (sz * shape[1] * (win_height - overlap_height),
                sz * (win_width - overlap_width), sz * shape[1], sz)
    shape = ((shape[0] - overlap_height) // (win_height - overlap_height), 
        (shape[1] - overlap_width) // (win_width - overlap_width), win_height, win_width)
    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, win_height, win_width)

def first_pass(frame_a, frame_b, win_width, win_height, overlap_width, overlap_height\
               ,iterations,correlation_method='circular', subpixel_method='gaussian'\
                   ,do_sig2noise=False, sig2noise_method='peak2peak', sig2noise_mask=2):
    """
    First pass of the PIV evaluation.
    This function does the PIV evaluation of the first pass. It returns
    the coordinates of the interrogation window centres, the displacment
    u and v for each interrogation window as well as the mask which indicates
    wether the displacement vector was interpolated or not.
    Parameters
    ----------
    frame_a : 2d np.ndarray
        the first image
    frame_b : 2d np.ndarray
        the second image
    window_size : int
         the size of the interrogation window
    overlap : int
        the overlap of the interrogation window normal for example window_size/2
    subpixel_method: string
        the method used for the subpixel interpolation.
        one of the following methods to estimate subpixel location of the peak:
        'centroid' [replaces default if correlation map is negative],
        'gaussian' [default if correlation map is positive],
        'parabolic'
    Returns
    -------
    x : 2d np.array
        array containg the x coordinates of the interrogation window centres
    y : 2d np.array
        array containg the y coordinates of the interrogation window centres 
    u : 2d np.array
        array containing the u displacement for every interrogation window
    u : 2d np.array
        array containing the u displacement for every interrogation window
    """
    
    cor_win_1 = moving_window_array_rectangular(frame_a, win_width, win_height,
                                                overlap_width, overlap_height)
    cor_win_2 = moving_window_array_rectangular(frame_b, win_width, win_height,
                                                overlap_width, overlap_height)
    '''Filling the interrogation window. They windows are arranged
    in a 3d array with number of interrogation window *window_size*window_size
    this way is much faster then using a loop'''

    correlation = correlation_func(cor_win_1, cor_win_2, win_width, win_height,
                                   correlation_method=correlation_method)
    'do the correlation'
    disp = np.zeros((np.size(correlation, 0), 2))#create a dummy for the loop to fill
    for i in range(0, np.size(correlation, 0)):
        ''' determine the displacment on subpixel level '''
        disp[i, :] = find_subpixel_peak_position(
            correlation[i, :, :], subpixel_method=subpixel_method)
    'this loop is doing the displacment evaluation for each window '

    shapes = get_field_shape(frame_a.shape, win_width, win_height, overlap_width, overlap_height)
    u = disp[:, 1].reshape(shapes)
    v = -disp[:, 0].reshape(shapes)
    'reshaping the interrogation window to vector field shape'
    
    x, y = get_coordinates(frame_a.shape, win_width, win_height, overlap_width, overlap_height)
    'get coordinates for to map the displacement'
    if do_sig2noise==True and iterations==1:
        sig2noise_ratio = sig2noise_ratio_function(correlation, sig2noise_method=sig2noise_method, width=sig2noise_mask)
        sig2noise_ratio = sig2noise_ratio.reshape(shapes)
    else:
        sig2noise_ratio=np.full_like(u,np.nan)
    return x, y, u, v, sig2noise_ratio

def multipass_img_deform(frame_a, frame_b, win_width, win_height, overlap_width, overlap_height,iterations,current_iteration,
                         x_old, y_old, u_old, v_old,correlation_method='circular',
                         subpixel_method='gaussian', do_sig2noise=False, sig2noise_method='peak2peak',
                         sig2noise_mask=2, MinMaxU=(-100, 50), MinMaxV=(-50, 50), std_threshold=5,
                         median_threshold=2,median_size=1, filter_method='localmean',
                         max_filter_iteration=10, filter_kernel_size=2, interpolation_order=3):
    """
    First pass of the PIV evaluation.
    This function does the PIV evaluation of the first pass. It returns
    the coordinates of the interrogation window centres, the displacment
    u and v for each interrogation window as well as the mask which indicates
    wether the displacement vector was interpolated or not.
    Parameters
    ----------
    frame_a : 2d np.ndarray
        the first image
    frame_b : 2d np.ndarray
        the second image
    window_size : tuple of ints
         the size of the interrogation window
    overlap : tuple of ints
        the overlap of the interrogation window normal for example window_size/2
    x_old : 2d np.ndarray
        the x coordinates of the vector field of the previous pass
    y_old : 2d np.ndarray
        the y coordinates of the vector field of the previous pass
    u_old : 2d np.ndarray
        the u displacement of the vector field of the previous pass
    v_old : 2d np.ndarray
        the v displacement of the vector field of the previous pass
    subpixel_method: string
        the method used for the subpixel interpolation.
        one of the following methods to estimate subpixel location of the peak:
        'centroid' [replaces default if correlation map is negative],
        'gaussian' [default if correlation map is positive],
        'parabolic'
    MinMaxU : two elements tuple
        sets the limits of the u displacment component
        Used for validation.
    MinMaxV : two elements tuple
        sets the limits of the v displacment component
        Used for validation.
    std_threshold : float
        sets the  threshold for the std validation
    median_threshold : float
        sets the threshold for the median validation
    filter_method : string
        the method used to replace the non-valid vectors
        Methods:
            'localmean',
            'disk',
            'distance', 
    max_filter_iteration : int
        maximum of filter iterations to replace nans
    filter_kernel_size : int
        size of the kernel used for the filtering
    interpolation_order : int
        the order of the spline interpolation used for the image deformation
    Returns
    -------
    x : 2d np.array
        array containg the x coordinates of the interrogation window centres
    y : 2d np.array
        array containg the y coordinates of the interrogation window centres 
    u : 2d np.array
        array containing the u displacement for every interrogation window
    u : 2d np.array
        array containing the u displacement for every interrogation window
    mask : 2d np.array
        array containg the mask values (bool) which contains information if
        the vector was filtered
    """

    x, y = get_coordinates(np.shape(frame_a), win_width, win_height, overlap_width, overlap_height)
    'calculate the y and y coordinates of the interrogation window centres'
    y_old = y_old[:, 0]
    y_old = y_old[::-1]
    x_old = x_old[0, :]
    y_int = y[:, 0]
    y_int = y_int[::-1]
    x_int = x[0, :]
    
    """
    This filters nans that might have been produced while invalid vectors are
    replaced. This is necessary because the initial ROI differs for different
    runs because it does not work for some, which is why the ROI is much larger
    than it should be, which produces nans in the beginning. Here we set them to 0.
    This is fine because in the beginning we are only interested in the bottom 5 rows
    """
    valid = np.isfinite(v_old)*np.isfinite(u_old)
    invalid = ~valid
    u_old[invalid] = 0
    v_old[invalid] = 0
    
    '''The interpolation function dont like meshgrids as input. Hence, the the edges
    must be extracted to provide the sufficient input. x_old and y_old are the 
    are the coordinates of the old grid. x_int and y_int are the coordiantes
    of the new grid'''
        
    ip = RectBivariateSpline(y_old, x_old, u_old)
    u_pre = ip(y_int, x_int)
    ip2 = RectBivariateSpline(y_old, x_old, v_old)
    v_pre = ip2(y_int, x_int)
    ''' interpolating the displacements from the old grid onto the new grid
    y befor x because of numpy works row major
    '''

    frame_b_deform = frame_interpolation(
        frame_b, x, y, u_pre, -v_pre, interpolation_order=interpolation_order)
    '''this one is doing the image deformation (see above)'''

    cor_win_1 = moving_window_array_rectangular(frame_a, win_width, win_height,
                                                overlap_width, overlap_height)
    cor_win_2 = moving_window_array_rectangular(frame_b_deform, win_width, win_height,
                                                overlap_width, overlap_height)
    '''Filling the interrogation window. They windows are arranged
    in a 3d array with number of interrogation window *window_size*window_size
    this way is much faster then using a loop'''

    correlation = correlation_func(cor_win_1, cor_win_2, win_width, win_height,
                                   correlation_method=correlation_method)
    'do the correlation'
    disp = np.zeros((np.size(correlation, 0), 2))
    for i in range(0, np.size(correlation, 0)):
        ''' determine the displacment on subpixel level  '''
        disp[i, :] = find_subpixel_peak_position(
            correlation[i, :, :], subpixel_method=subpixel_method)
    'this loop is doing the displacment evaluation for each window '

    'reshaping the interrogation window to vector field shape'
    shapes = get_field_shape(frame_a.shape, win_width, win_height, overlap_width, overlap_height)
    u = disp[:, 1].reshape(shapes)
    v = -disp[:, 0].reshape(shapes)

    'adding the recent displacment on to the displacment of the previous pass'
    u = u+u_pre
    v = v+v_pre
    'validation using gloabl limits and local median'
    u, v, mask_g = validation.global_val(u, v, MinMaxU, MinMaxV)
    u, v, mask_s = validation.global_std(u, v, std_threshold=std_threshold)
    u, v, mask_m = validation.local_median_val(u, v, u_threshold=median_threshold,
                                               v_threshold=median_threshold, size=median_size)
    mask = mask_g+mask_m+mask_s
    'adding masks to add the effect of alle the validations'
    #mask=np.zeros_like(u)
    'filter to replace the values that where marked by the validation'
    if current_iteration != iterations:
        'filter to replace the values that where marked by the validation'
        u, v = filters.replace_outliers(
                    u, v, method=filter_method, max_iter=max_filter_iteration,
                    kernel_size=filter_kernel_size) 
    if do_sig2noise==True and current_iteration==iterations and iterations!=1:
        sig2noise_ratio=sig2noise_ratio_function(correlation, sig2noise_method=sig2noise_method, width=sig2noise_mask)
        sig2noise_ratio = sig2noise_ratio.reshape(shapes)
        np.save('corr_map_valid.npy', correlation[423,:,:])
        np.save('corr_map_invalid.npy', correlation[50,:,:])
    else:
        sig2noise_ratio=np.full_like(u,np.nan)

    

    return x, y, u, v,sig2noise_ratio, mask


def save( x, y, u, v, sig2noise_ratio, mask, filename, fmt='%8.4f', delimiter='\t' ):
    """Save flow field to an ascii file.
    
    Parameters
    ----------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the 
        interrogation window centers, in pixels.
        
    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the 
        interrogation window centers, in pixels.
        
    u : 2d np.ndarray
        a two dimensional array containing the u velocity components,
        in pixels/seconds.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity components,
        in pixels/seconds.
        
    mask : 2d np.ndarray
        a two dimensional boolen array where elements corresponding to
        invalid vectors are True.
        
    filename : string
        the path of the file where to save the flow field
        
    fmt : string
        a format string. See documentation of numpy.savetxt
        for more details.
    
    delimiter : string
        character separating columns
        
    Examples
    --------
    
    >>> openpiv.tools.save( x, y, u, v, 'field_001.txt', fmt='%6.3f', delimiter='\t')
    
    """
    # build output array
    out = np.vstack( [m.ravel() for m in [x, y, u, v,sig2noise_ratio, mask] ] )
            
    # save data to file.
    np.savetxt( filename, out.T, fmt=fmt, delimiter=delimiter, header='x'+delimiter+'y'+delimiter+'u'+delimiter+'v'+delimiter+'s2n'+delimiter+'mask' )
    
def display_vector_field( filename, on_img=False, image_name='None', window_size=32, scaling_factor=1,skiprows=1, **kw):
    """ Displays quiver plot of the data stored in the file 
    
    
    Parameters
    ----------
    filename :  string
        the absolute path of the text file
    on_img : Bool, optional
        if True, display the vector field on top of the image provided by image_name
    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True
    window_size : int, optional
        when on_img is True, provide the interogation window size to fit the background image to the vector field
    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background image to the vector field
    
    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]
    
    
    See also:
    ---------
    matplotlib.pyplot.quiver
    
        
    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100, width=0.0025) 
    --- vector field on top of image
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt', on_img=True, image_name='exp1_001_a.bmp', window_size=32, scaling_factor=70, scale=100, width=0.0025)
    
    """
    
    a = np.loadtxt(filename)
    fig=plt.figure()
    if on_img: # plot a background image
        im = fig.imread(image_name)
        im = fig.negative(im) #plot negative of the image for more clarity
        fig.imsave('neg.tif', im)
        im = fig.imread('neg.tif')
        xmax = np.amax(a[:,0])+window_size/(2*scaling_factor)
        ymax = np.amax(a[:,1])+window_size/(2*scaling_factor)
        implot = plt.imshow(im, origin='lower', cmap="Greys_r",extent=[0.,xmax,0.,ymax])
    invalid = a[:,5].astype('bool')
    fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')
    valid = ~invalid
    plt.quiver(a[invalid,0],a[invalid,1],a[invalid,2],a[invalid,3],color='r',width=0.001,headwidth=3,**kw)
    plt.quiver(a[valid,0],a[valid,1],a[valid,2],a[valid,3],color='b',width=0.001,headwidth=3,**kw)
    plt.draw()


def get_field_shape(image_size, win_width, win_height, overlap_width, overlap_height):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    
    Manuel: Modified to include the different widths and heights of the image
    
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



def get_coordinates(image_size, win_width, win_height, overlap_width, overlap_height):
        """Compute the x, y coordinates of the centers of the interrogation windows.
        
        Manuel: Added the different overlaps and widths and heights for
            rectangular windows
        
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
        peak1_i, peak1_j, dummy = pyprocess.find_first_peak(corr)
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
    


def sig2noise_ratio_function(corr, sig2noise_method='peak2peak', width=2):
    """
    Computes the signal to noise ratio from the correlation map.
    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interogation windows.
    Parameters
    ----------
    corr : 2d np.ndarray
        the correlation map.
    sig2noise_method: string
        the method for evaluating the signal to noise ratio value from
        the correlation map. Can be `peak2peak`, `peak2mean` or None
        if no evaluation should be made.
    width : int, optional
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
    Returns
    -------
    sig2noise : np.ndarray 
        the signal to noise ratio from the correlation map.
        
    Manuel: 
        Implemented the possibility to do sig2RMS, which divides the maximum
        by the standard deviation of the correlation map. Also changed the 
        calculation for peak2mean, which makes it faster
    """

    corr_max1=np.zeros(corr.shape[0])
    corr_max2=np.zeros(corr.shape[0])
    peak1_i=np.zeros(corr.shape[0])
    peak1_j=np.zeros(corr.shape[0])
    if sig2noise_method == 'peak2peak':
        peak2_i=np.zeros(corr.shape[0])
        peak2_j = np.zeros(corr.shape[0])
        for i in range(0,corr.shape[0]):
            # compute first peak position
            peak1_i[i], peak1_j[i], corr_max1[i] = pyprocess.find_first_peak(corr[i,:,:])
            
            # now compute signal to noise ratio
            
            # find second peak height
            peak2_i[i], peak2_j[i], corr_max2[i] = pyprocess.find_second_peak(
                corr[i,:,:], int(peak1_i[i]), int(peak1_j[i]), width=width)
    
            # if it's an empty interrogation window
            # if the image is lacking particles, totally black it will correlate to very low value, but not zero
            # if the first peak is on the borders, the correlation map is also
            # wrong
            if corr_max1[i] < 1e-3 or (peak1_i[i] == 0 or peak1_j[i] == corr.shape[1] or peak1_j[i] == 0 or peak1_j[i] == corr.shape[2] or
                                    peak2_i[i] == 0 or peak2_j[i] == corr.shape[1] or peak2_j[i] == 0 or peak2_j[i] == corr.shape[2]):
                # return zero, since we have no signal.
                corr_max1[i]=0
    
    elif sig2noise_method == 'peak2mean':
        for i in range(0,corr.shape[0]):
            peak1_i[i], peak1_j[i], corr_max1[i] = pyprocess.find_first_peak(corr[i,:,:])
        # find mean of the correlation map
        corr_max2 = corr.mean(axis=(1,2))
    elif sig2noise_method == 'peak2RMS':
        for i in range(0,corr.shape[0]):
            peak1_i[i], peak1_j[i], corr_max1[i] = pyprocess.find_first_peak(corr[i,:,:])            
        # find the RMS of the correlation map
        corr_max2 = corr.std(axis=(1,2))
    else:
        raise ValueError('wrong sig2noise_method')

    # avoid dividing by zero
    corr_max2[corr_max2==0]=np.nan    
    sig2noise = corr_max1 / corr_max2
    sig2noise[sig2noise==np.nan]=0

    return sig2noise



class Settings(object):
    pass


if __name__ == "__main__":
    """ Run windef.py as a script: 
    python windef.py 
    """



    settings = Settings()


    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = './examples/test1/'
    # Folder for the outputs
    settings.save_path = './examples/test1/'
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_4'
    # Format and Image Sequence
    settings.frame_pattern_a = 'exp1_001_a.bmp'
    settings.frame_pattern_b = 'exp1_001_b.bmp'

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.ROI = 'full'

    'Image preprocessing'
    # 'None' for no masking, 'edges' for edges masking, 'intensity' for intensity masking
    # WARNING: This part is under development so better not to use MASKS
    settings.dynamic_masking_method = 'None'
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7

    'Processing Parameters'
    settings.correlation_method='circular'  # 'circular' or 'linear'
    settings.iterations =1  # select the number of PIV passes
    # add the interroagtion window size for each pass. 
    # For the moment, it should be a power of 2 
    settings.windowsizes = (128, 64, 32) # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    settings.overlap = (64, 32, 16) # This is 50% overlap
    # Has to be a value with base two. In general window size/2 is a good choice.
    # methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
    settings.subpixel_method = 'gaussian'
    # order of the image interpolation for the window deformation
    settings.interpolation_order = 3
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)
    'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = 'peak2peak'
    # select the width of the masked to masked out pixels next to the main peak
    settings.sig2noise_mask = 2
    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN
    'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True
    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated
    'Validation Parameters'
    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    settings.MinMax_U_disp = (-30, 30)
    settings.MinMax_V_disp = (-30, 30)
    # The second filter is based on the global STD threshold
    settings.std_threshold = 10  # threshold of the std validation
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size=1 #defines the size of the local median
    'Validation based on the signal to noise ratio'
    # Note: only available when extract_sig2noise==True and only for the last
    # pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    settings.do_sig2noise_validation = False # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    settings.sig2noise_threshold = 1.2
    'Outlier replacement or Smoothing options'
    # Replacment options for vectors which are masked as invalid by the validation
    settings.replace_vectors = True # Enable the replacment. Chosse: True or False
    settings.smoothn=True #Enables smoothing of the displacemenet field
    settings.smoothn_p=0.5 # This is a smoothing parameter
    # select a method to replace the outliers: 'localmean', 'disk', 'distance'
    settings.filter_method = 'localmean'
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method
    'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = True
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = False
    settings.scale_plot = 100 # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings

    piv(settings)