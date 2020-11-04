# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:37:58 2019

@author: mendez, torres
"""

import os  # This is to understand which separator in the paths (/ or \)
import numpy as np  # This is for doing math
import cv2

def get_image_params(folder):
    """
    Function to get the filename and the first index
    :param folder:
    """
    # get a list of the images
    image_list = os.listdir(folder)
    # take the name of the first one
    name0 = image_list[0]
    # search the '.' in the file name
    indices = [i for i, a in enumerate(name0) if a == '.']
    # crop the name to get the beginning
    nomenclature = name0[:indices[0]]
    # extract the first image
    idx0 = int(name0[indices[0]+1:indices[1]])
    return nomenclature, idx0

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

def export_images(matrix, folder, n_images, shape):
    """
    Function to export the images
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

# Folder in
Fol_In = 'C:\PIV_Processed\Images_Rotated\F_h4_f1200_1_q' + os.sep
# create output folder in case it doesn't exist
Fol_Out = 'C:\PIV_Processed\Images_Preprocessed\F_h4_f1200_1_q' + os.sep  
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

# get the name of the images and the first index
image_name, idx0 = get_image_params(Fol_In)
# load the first image to get the shape of the images
Name=generate_filename(Fol_In, image_name, idx0, 'tif')
Im = cv2.imread(Name, 0)
ny, nx = Im.shape

# load all the image names in the directory and initialize the data matrix
img_list = os.listdir(Fol_In)
n_t = len(img_list)
# n_t = 1000; idx0 = 0
D_a = np.zeros((nx * ny, n_t))

# loop over all the images
for k in range(0, n_t):
    # prepare the image for the matrix
    ImV = process_image_for_matrix_D(FOL_IN=Fol_In, iter=(k+idx0))
    # update the user
    print('Loading ' + str(k+1) + '/' + str(n_t))
    # append into matrix
    D_a[:, k] = ImV[:, 0]


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
export_images(D_a_filt, Fol_Out, n_images=n_t, shape=(ny, nx))
