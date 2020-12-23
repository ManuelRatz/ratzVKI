### Readme file for the processing of the PIV campaign
This folder contains the codes to process the piv campaign, it is structured as follows:
* 0_Processing_Federica - folder containing the files to process images for Federica. The settings have been extended to 
also include rectangular windows for images with low seeding
* 1_Pre_Processing - folder containting codes to preprocess the raw images, this includes cropping and rotating as well
the POD to remove the background noise
* 2_1_Fall_processing - folder containing the codes to process the fall of the images using a moving roi when
the interface comes into the FOV
* 2_2_Rise_processing - folder containing the codes to process the rise of the images using a moving roi when
the interface comes into the FOV
* 3_Post_Processing.py - folder containing the codes to smooth the velocity profiles in 3 dimensions and animate 
the result for the giant slide show containing every test case
* post_processing_functions - library containing every possible function used for the post processing
* smoothn - smoothn function by Damien Garcia, used in the post processing