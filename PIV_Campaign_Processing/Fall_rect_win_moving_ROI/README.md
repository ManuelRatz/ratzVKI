### Readme file for the processing of the falls
Files to process the fall of the piv images. To process them, run *2_OpenPiv_windef_client_rect_moving_roi.py*.
The code will loop over the folders containting all the fall measurements conducted. It is possible to process two things:
1. The fall with the full ROI, for this enable settings.process_fall
1. The fall with just the moving ROI, for this enable settings.process_roi_shift

The file stores the images in the specified folder. The name for the folder is taken from *observation_fall.txt*.
This file contains the name of the runs and also the indices of the images where the fall starts and when the interface comes
into the FOV. 

Additional files and folders:
* test_images - folder containing a few test images, the input in the client must be modified for this to work
* smoothn, tools_patch, tools_patch_fall, windef_rect_moving_roi_fall - files for the processing of the piv
* moving_roi_animation - file to animate the moving roi for each run
* fall_post_processing - post processing file for animating the results