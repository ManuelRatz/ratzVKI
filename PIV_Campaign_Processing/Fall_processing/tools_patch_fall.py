#!/usr/bin/env python
"""The openpiv.tools module is a collection of utilities and tools.
"""

__licence__ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Modifications by Manuel:
    Used cv2 for reading the images because sometimes errors popped up
    Changed the Multiprocessor to include the index of the first image
"""
import glob
import os
import numpy as np
import matplotlib.image as ig
import matplotlib.pyplot as pl
import matplotlib.patches as pt

def save_windef( x, y, u, v, sig2noise_ratio, mask, filename, fmt='%8.4f', delimiter='\t' ):
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
    
def display_vector_field_windef( filename, on_img=False, image_name='None', window_size=32, scaling_factor=1, widim = False, ax = None, **kw):
    """ Displays quiver plot of the data stored in the file 
     #MYEDIT
    
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
    if ax is None: #MYEDIT
        fig, ax = pl.subplots() #MYEDIT
    else: #MYEDIT
        fig = ax.get_figure() #MYEDIT
    if on_img: # plot a background image
        im = ig.imread(image_name)
        im = ig.negative(im) #plot negative of the image for more clarity
        #ig.imsave('neg.tif', im)
        #im = ig.imread('neg.tif') MYEDIT
        xmax=np.amax(a[:,0])+window_size/(2*scaling_factor)
        ymax=np.amax(a[:,1])+window_size/(2*scaling_factor)
        ax.imshow(im, origin='lower', cmap="Greys_r", #MYEDIT
                  extent=[0., xmax, 0., ymax])
        pl.draw() #MYEDIT
    invalid = a[:,5].astype('bool')
    #MYEDIT fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')
    valid = ~invalid
    pl.quiver(a[invalid,0],a[invalid,1],a[invalid,2],a[invalid,3],color='r',width=0.001,headwidth=3,**kw)
    pl.quiver(a[valid,0],a[valid,1],a[valid,2],a[valid,3],color='b',width=0.001,headwidth=3,**kw)
    # pl.show()
    return fig, ax

def imread(filename, flatten=0):
    """Read an image file into a numpy array
    using imageio.imread 
    
    Parameters
    ----------
    filename :  string
        the absolute path of the image file
    flatten :   bool
        True if the image is RGB color or False (default) if greyscale
        
    Returns
    -------
    frame : np.ndarray
        a numpy array with grey levels
        
        
    Examples
    --------
    
    >>> image = openpiv.tools.imread( 'image.bmp' )
    >>> print image.shape 
        (1280, 1024)
    
    
    """
    import cv2
    im = cv2.imread(filename,0)
    # im = _imread(filename)
    if np.ndim(im) > 2:
        im = rgb2gray(im)

    return im

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])



class Multiprocesser():
    def __init__ ( self, data_dir, pattern_a, pattern_b = None):
        """A class to handle and process large sets of images.

        This class is responsible of loading image datasets
        and processing them. It has parallelization facilities
        to speed up the computation on multicore machines.
        
        It currently support only image pair obtained from 
        conventional double pulse piv acquisition. Support 
        for continuos time resolved piv acquistion is in the 
        future.
        
        
        Parameters
        ----------
        data_dir : str
            the path where image files are located 
            
        pattern_a : str
            a shell glob patter to match the first 
            frames.
            
        pattern_b : str
            a shell glob patter to match the second
            frames. if None, then the list is sequential, 001.tif, 002.tif 

        Examples
        --------
        >>> multi = openpiv.tools.Multiprocesser( '/home/user/images', 'image_*_a.bmp', 'image_*_b.bmp')
    
        """
        # load lists of images
         
        self.files_a = sorted( glob.glob( os.path.join( os.path.abspath(data_dir), pattern_a ) ) )
        indices = [i for i, a in enumerate(self.files_a[0][::-1]) if a == '.']
        self.index_0 = int(self.files_a[0][len(self.files_a[0])-indices[1]:len(self.files_a[0])-indices[0]-1])
        if pattern_b is None:
            self.files_b = self.files_a[1:]
            self.files_a = self.files_a[:-1]
        else:    
            self.files_b = sorted( glob.glob( os.path.join( os.path.abspath(data_dir), pattern_b ) ) )
        
        # number of images
        self.n_files = len(self.files_a)
        # check if everything was fine
        if not len(self.files_a) == len(self.files_b):
            raise ValueError('Something failed loading the image file. There should be an equal number of "a" and "b" files.')
            
        if not len(self.files_a):
            raise ValueError('Something failed loading the image file. No images were found. Please check directory and image template name.')

    def run( self, save_path, func, fall_start, roi_shift_start, process_fall, process_roi_shift, n_cpus=1 ):
        """Start to process images.
        
        Parameters
        ----------
        
        func : python function which will be executed for each 
            image pair. See tutorial for more details.
        
        n_cpus : int
            the number of processes to launch in parallel.
            For debugging purposes use n_cpus=1
        
        """
        if (process_fall == False and process_roi_shift == False):
            raise ValueError('No images are being processed as both calculations have been disabled')
        if process_fall == True:
            beginning_index = fall_start
        else:
            beginning_index = roi_shift_start
        # create a list of tasks to be executed.
        image_pairs = [ (file_a, file_b, i+beginning_index)\
                       for file_a, file_b, i in zip( self.files_a[beginning_index-self.index_0:],\
                                                    self.files_b[beginning_index-self.index_0:], range(self.n_files)) ]
        index_max = beginning_index+self.n_files
        h_dum = np.zeros((index_max,1))
        # for debugging purposes always use n_cpus = 1,
        # since it is difficult to debug multiprocessing stuff.
        for image_pair in image_pairs:
            # this is to check whether we have to stop because the roi reached the lowest possible point
            # if(image_pair[2] < 709):
            #     continue
            h, stop_iteration = func(image_pair)
            h_dum[image_pair[2]-1] = h
            if (stop_iteration == True):
                break
        np.savetxt(save_path + os.sep + 'interface_position.txt', h_dum, fmt='%8.4f')