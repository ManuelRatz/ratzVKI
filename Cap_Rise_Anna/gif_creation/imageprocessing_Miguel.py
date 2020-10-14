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

F=np.array([[1,0,0,1],[2,1,0,3],[3,2,1,1],[2,2,2,1]])/4
name = 'test_images' + os.sep + '350Pa_C01257.png'
im = cv2.imread(name,0)
im_float = im.astype(np.float)

fig=plt.imshow(im, cmap=plt.cm.gray)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.imshow(fig)

# # Part 2: Equalization
# img_eq = exposure.equalize_hist(im_float)
# plt.imshow(im, cmap=plt.cm.gray)
# fig.axes.get_xaxis().set_visible(False)
# fig.axes.get_yaxis().set_visible(False)
# plt.show(fig)

# Part 1: Gamma Correction

img_gamma=exposure.adjust_gamma(im,gamma=2,gain=1)
# print(img_gamma)
fig=plt.imshow(im, cmap=plt.cm.gray)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)