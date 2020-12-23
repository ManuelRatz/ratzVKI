# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:45:42 2020

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import cv2

Index = 309
Fol_Raw = 'C:\PIV_Processed\Images_Rotated\F_h2_f1000_1_q' + os.sep
Fol_POD = 'C:\PIV_Processed\Images_Preprocessed\F_h2_f1000_1_q' + os.sep
Img_name = Fol_Raw + 'F_h2_f1000_1_q.%06d.tif' %Index 
img = cv2.imread(Img_name,0)

Img_name_pod = Fol_POD + 'F_h2_f1000_1_q.%06d.tif' %Index 
img_pod = cv2.imread(Img_name_pod,0)


blurred1 = ndimage.gaussian_filter(img, sigma = 4, mode = 'nearest', truncate = 5)
filt = np.abs(img.astype(np.int32) - blurred1.astype(np.int32))
# filt = np.fliplr(filt)
# filt = np.fliplr(blurred)
filt = (filt/np.max(filt)*255).astype(np.uint8)

output = np.hstack((img, blurred1, filt, np.fliplr(img_pod)))
output = output[0:400]
plt.imshow(output, cmap = plt.cm.gray)
plt.axis('off')
plt.savefig('High_pass_test.png',dpi = 600)