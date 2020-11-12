# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:08:32 2020

@author: manue
"""

import cv2
import imageio
import os

Fol_In = 'D:\PIV_Processed\Images_Processed'
Fol_Out = 'D:\PIV_Processed\Animations'

folders = os.listdir(Fol_In)

for i in range(0, len(folders)):
# for i in range(0, 1):
    Image_location = Fol_In + os.sep + folders[i] + os.sep + 'ROI_images' + os.sep
    images = os.listdir(Image_location)
    Gif_Name = Fol_Out + os.sep + folders[i] + '.gif'
    gif_images = []
    for j in range(0, len(images)):
    # for j in range(0, 5):
        gif_images.append(cv2.imread(Image_location + images[j],0))
    imageio.mimsave(Gif_Name, gif_images, duration=0.05)
    print('Finished %s' %folders[i])
                