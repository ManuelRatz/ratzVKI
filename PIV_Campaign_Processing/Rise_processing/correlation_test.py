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

data = np.load('Image_test.npz')
Image_Test=data['Image']

win_height = 256
win_width = 64

cor_win_1 = Image_Test
cor_win_1 = cor_win_1-np.mean(cor_win_1)

# still under development

# corr = fftshift(irfft2(np.conj(rfft2(cor_win_1)) *
                          # rfft2(cor_win_1)).real) # change back later
corr = fftshift(irfft2(np.conj(rfft2(cor_win_1,s=(2*win_height,2*win_width))) *
                          rfft2(cor_win_1,s=(2*win_height,2*win_width))).real)
corr=corr[win_height//2:3*win_height//2,win_width//2:3*win_width//2]
std_a = np.std(cor_win_1)

# corr_norm = np.zeros(corr.shape)
corr_norm = corr/(std_a**2*64*256)
corr_norm[corr_norm<0]=0
# corr_norm = corr/(std_a*std_b)