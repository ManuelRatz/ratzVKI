# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:17:12 2020

@author: Manuel Ratz
@description: Code to cut the ending given by the dantec files.
    Due to the different nomenclature we take the same loop twice,
    once for the fall and once for the rise
"""
import os # for data paths and renaming

# Fol_In = 'C:'+os.sep+'PIV_Campaign'+os.sep+'Rise'+os.sep # directory containing the heights
# heights = os.listdir(Fol_In) # list of all the heights
# for m in range(0,len(heights)): # loop over them
#     pressures = os.listdir(Fol_In+heights[m]) # get the pressure for each of the heights
#     for i in range(0, len(pressures)): # loop over them
#         runs = os.listdir(Fol_In+heights[m]+os.sep + pressures[i]) # get the different runs done for this pressure
#         for j in range(0, len(runs)): # loop over them
#             current_folder = Fol_In+heights[m]+os.sep + pressures[i]+os.sep+runs[j]+os.sep # folder name containing images
#             img_list = os.listdir(current_folder) # list of the images and lvm files
#             n_t = len(img_list)-5 # get the image amount by subtracting the lvm files
#             if n_t <6: # check whether there are images in the directory
#                 print('No images in directory %s' %current_folder) # error message
#                 continue
#             name = img_list[5] # get the name of the first image
            
#             indices = [i for i, a in enumerate(name) if a == '.'] # find all the '.' in the name
#             if (len(indices)==3): # check whether name still has all 3 dots
#                 name_old = name[:indices[1]+1] # get the old name without tif ending
#                 name_new = name[:indices[0]+1] # get the new name without the dantec ending
#             else: # error message
#                 print('Images in Directory %s have false name' %current_folder)
#                 continue
            
#             print('Renaming folder %s' %current_folder) # update user
#             for k in range(0,n_t): # loop over all the images
#                 old_directory = current_folder + name_old + '%06d.tif' %k # old name of the image
#                 new_directory = current_folder + name_new + '%06d.tif' %k # new name of the image
#                 os.rename(old_directory, new_directory) # rename the image
            
# Fol_In = 'C:'+os.sep+'PIV_Campaign'+os.sep+'Fall'+os.sep # directory containing the heights
# heights = os.listdir(Fol_In) # list of all the heights
# for m in range(0,len(heights)): # loop over them
#     speeds = os.listdir(Fol_In+heights[m])
#     for j in range(0,len(speeds)):
#         current_folder = Fol_In+heights[m]+os.sep+speeds[j]+os.sep
#         img_list = os.listdir(current_folder) # list of the images and lvm files
#         n_t = len(img_list)-5 # get the image amount by subtracting the lvm files
#         if n_t <6: # check whether there are images in the directory
#             print('No images in directory %s' %current_folder) # error message
#             continue
#         name = img_list[5] # get the name of the first image
#         indices = [i for i, a in enumerate(name) if a == '.'] # find all the '.' in the name
#         if (len(indices)==3): # check whether name still has all 3 dots
#             name_old = name[:indices[1]+1] # get the old name without tif ending
#             name_new = name[:indices[0]+1] # get the new name without the dantec ending
#         else: # error message
#             print('Images in Directory %s are renamed or not named properly' %current_folder)
#             continue
#         first_idx = int(name[indices[1]+1:indices[2]])
#         print('Renaming folder %s' %current_folder) # update user
#         for k in range(first_idx,first_idx+n_t): # loop over all the images
#             old_directory = current_folder + name_old + '%06d.tif' %k # old name of the image
#             new_directory = current_folder + name_new + '%06d.tif' %k # new name of the image
#             os.rename(old_directory, new_directory) # rename the image

"""
These are the runs that have been exported later because of different issues
"""

# names of the missing runs
runs = ['F_h2_f1000_1_s', 'R_h1_f750_1_p10', 'R_h1_f1200_1_p14']
# iterate over them
for run in runs:
    # give the location
    current_folder = 'C:\PIV_Campaign' + os.sep + run + os.sep
    # get the list of images
    img_list = os.listdir(current_folder)
    # take the first frame (these dont have the LVM files)
    name = img_list[0]
    # get the '.' out of the name
    indices = [i for i, a in enumerate(name) if a == '.']
    if (len(indices)==3): # check whether name still has all 3 dots
        name_old = name[:indices[1]+1] # get the old name without tif ending
        name_new = name[:indices[0]+1] # get the new name without the dantec ending
    # get the first index
    first_idx = int(name[indices[1]+1:indices[2]])
    # get the number of images
    n_t = len(img_list)
    # iterate over them
    for k in range(first_idx,first_idx+n_t): # loop over all the images
        old_directory = current_folder + name_old + '%06d.tif' %k # old name of the image
        new_directory = current_folder + name_new + '%06d.tif' %k # new name of the image
        os.rename(old_directory, new_directory) # rename the image