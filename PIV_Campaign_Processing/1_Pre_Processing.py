# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:37:58 2019

@author: mendez, torres
"""

import os  # This is to understand which separator in the paths (/ or \)
import numpy as np  # This is for doing math
import cv2

def get_image_params(Fol_In):
    image_list = os.listdir(Fol_In)
    name0 = image_list[0]
    indices = [i for i, a in enumerate(name0) if a == '.']
    ret = name0[:indices[0]]
    idx0 = int(name0[indices[0]+1:indices[1]])
    return ret, idx0

def generate_filename(folder, image_name, number, pic_format):
    """
    simple function to generate filenames
    :param folder: where to look or save
    :param number: number of the picture
    :param pair: a or b
    :param pic_format: format of the image
    :return:
    """
    return folder + os.sep + image_name+ '.%06d' % (number) + '.' + pic_format

# Folder in
Fol_In = 'C:\PIV_Processed\Images_Rotated\F_h2_f1000_1_q' + os.sep
# Processing Images
Fol_Out = 'C:\PIV_Processed\Images_Preprocessed\F_h2_f1000_1_q' + os.sep  # Where will the result be
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

image_name, idx0 = get_image_params(Fol_In)

# To define the crop area you will usually need to load at least one image 
# and plot on it a rectangle
Name=generate_filename(Fol_In, image_name, idx0, 'tif')
Im = cv2.imread(Name, 0)
ny, nx = Im.shape

# We can now proceed with the removal of the first POD mode
# This is a good idea only if you have a lot of images; otherwise limit yourself to teh
# Fourier filtering only.
# In this case just comment from the lines ------
# Comment from here to remove the POD processing------------------------
########################  POD - 1 Mode Removal ##########################

def process_image_for_matrix_D(FOL_IN, iter):
    """
    Function to process image
    :param FOL_IN: str folder in (for filename)
    :param iter: number of iteration (for filename)
    :param pair: str 'a' or 'b' (for filename)
    :return: np.float64 array
    """
    name = generate_filename(FOL_IN, image_name, iter, pic_format="tif")  # Check it out: print(Name)
    Im = cv2.imread(name,0)  # Read the image as uint8 using mpimg
    Imd = np.float64(Im)  # We work with floating number not integers
    ImV = np.reshape(Imd, ((nx * ny, 1)))  # Reshape into a column Vector
    return ImV



img_list = os.listdir(Fol_In)
n_t = len(img_list)
D_a = np.zeros((nx * ny, n_t))  # Initialize the Data matrix for image sequences A.

for k in range(0, n_t):
    # Prepare the Matrix D_a
    ImV = process_image_for_matrix_D(FOL_IN=Fol_In, iter=(k+idx0))
    print('Loading ' + str(k+1) + '/' + str(n_t))  # Print a Message to update the user
    D_a[:, k] = ImV[:, 0]


################ Computing the Filtered Matrices##########################
Ind_S = 1  # Number of modes to remove. If 0, the filter is not active!
# Compute the correlation matrix
print('Computing Correlation Matrices')
K_a = np.dot(D_a.transpose(), D_a)
print('K_a Ready')
# Comput the Temporal basis for A
Psi, Lambda, _ = np.linalg.svd(K_a)
# Compute the Projection Matrix
PSI_CROP = Psi[:, Ind_S::]
PROJ = np.dot(PSI_CROP, PSI_CROP.transpose())
D_a_filt = np.dot(D_a, PROJ)
print('D_a Filt Ready')

# Prepare Exporting the images

def export_images(matrix, folder, n_images, shape):
    """

    :param matrix: np.array matrix to extract the images
    :param folder: str folder out
    :param n_images: int number of images
    :param pair: str "a" or "b"
    :param shape: tuple
    """
    (ny, nx) = shape
    for k in range(0, n_images):
        name = generate_filename(folder, image_name, k+idx0, pic_format="tif")  # Check it out: print(Name)
        print('Exporting %i'%(k+1))
        Imd_V = matrix[:, k]
        Im = np.reshape(Imd_V, ((ny, nx)))
        Im[Im < 0] = 0  # Things below 0 are treated as zero
        Im2 = np.uint8(Im)
        cv2.imwrite(name, Im2)


export_images(D_a_filt, Fol_Out, n_images=n_t, shape=(ny, nx))
