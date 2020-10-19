# -- coding: utf-8 --
"""
Created on Sat Oct 3 12:38:58 2020

@author: mendez
"""

import numpy as np
import os
import cv2
# from scipy import signal

from skimage import exposure # Toolbox of Intensity Transforms

from matplotlib import pyplot as plt
crop_index = (66, 210, 0, 1280) 

for i in range(0,1):
    name = 'test_images' + os.sep + 'raw_images' + os.sep + '350Pa_C%05d.png' %(1460+20*i)
    im = cv2.imread(name,0)
    # initial crop
    im = im[crop_index[2]:crop_index[3],crop_index[0]:crop_index[1]]    
    denoise = 1
    dst = cv2.fastNlMeansDenoising(im,denoise,denoise,7,21)
    # dst = dst[540:640,0:144]
    # # # Part 2: Equalization
    img_eq = exposure.equalize_hist(dst)
    img_eq = (img_eq*20).astype(np.uint8)
    # img_gamma=exposure.adjust_gamma(img_eq,gamma=10,gain=1)
     
    img_gamma = img_eq[540:640,0:144]
    fig=plt.imshow(img_gamma, cmap=plt.cm.gray)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('test.png',dpi=400)
    plt.show(fig)
    
    
    # img_gamma=exposure.adjust_gamma(im,gamma=2,gain=1)
    # crop
    # im = img_eq[400:700,0:144] 
    # print(img_gamma)