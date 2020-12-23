### Readme file for the processing of the falls
Files to process the fall of the piv images. To process them, run *client_fall.py*.
The code will loop over the folders containting all the fall measurements conducted.

The file stores the images in the specified folder. The name for the folder is taken from *observation_fall.txt*.
This file contains the name of the runs and also the indices of the images where the fall starts and when the interface comes
into the FOV. 

Additional files and folders:
* smoothn, tools_patch, tools_patch_fall, validation_patch, windef_rect_fall - files for the processing of the piv