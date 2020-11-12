### Readme file for the processing of the PIV campaign
This folder contains the codes to process the piv campaign, it is structured as follows:
* Fall_rect_win_moving_ROI - folder containing the codes to process the fall of the images using a moving roi when
the interface comes into the FOV
* Pre_Processing - folder containting codes to preprocess the raw images, this includes cropping and rotating as well
the POD to remove the background noise
* Processing_Federica - folder containing the files to process images for Federica. The settings have been extended to 
also include rectangular windows for images with low seeding
* 3_Post_Processing.py - nothing added yet, still under development
* PressureLVM.py - code to process the pressure signals from the LabView files, not used currently