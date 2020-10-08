from OpenPIV_windef_func import PIV_windef

class Settings(object):
    pass  

settings = Settings()

'Data related settings'
# Folder with the images to process
settings.filepath_images = './Pre_Pro_PIV_IMAGES/'
# Folder for the outputs
settings.save_path = './Results_PIV/'
# Root name of the output Folder for Result Files
settings.save_folder_suffix = 'Test_1'
# Format and Image Sequence
settings.frame_pattern_a = 'Cyl_R_150_k7_001a.tif'
settings.frame_pattern_b = 'Cyl_R_150_k7_001b.tif'    

'Region of interest'
# (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
settings.ROI = 'full'

'Image preprocessing'
# 'None' for no masking, 'edges' for edges masking, 'intensity' for intensity masking
# WARNING: This part is under development so better not to use MASKS
settings.dynamic_masking_method = 'intensity'
settings.dynamic_masking_threshold = 0.005
settings.dynamic_masking_filter_size = 7 

'Processing Parameters'
settings.correlation_method = 'circular'  # 'circular' or 'linear'
settings.iterations = 2  # select the number of PIV passes
# add the interroagtion window size for each pass. 
# For the moment, it should be a power of 2 
settings.windowsizes = (128, 64) # if longer than n iteration the rest is ignored
# The overlap of the interroagtion window for each pass.
settings.overlap = (64, 32)  # This is 50% overlap
# Has to be a value with base two. In general window size/2 is a good choice.
# methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
settings.subpixel_method = 'gaussian'
# order of the image interpolation for the window deformation
settings.interpolation_order = 5
settings.scaling_factor = 1000  # scaling factor pixel/meter
settings.dt = 0.5  # time between to frames (in seconds)
'Signal to noise ratio options (only for the last pass)'
# It is possible to decide if the S/N should be computed (for the last pass) or not
settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
# method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
settings.sig2noise_method = 'peak2peak'
# select the width of the masked to masked out pixels next to the main peak
settings.sig2noise_mask = 2
# If extract_sig2noise==False the values in the signal to noise ratio
# output column are set to NaN
'vector validation options'
# choose if you want to do validation of the first pass: True or False
settings.validation_first_pass = True
# only effecting the first pass of the interrogation the following passes
# in the multipass will be validated
'Validation Parameters'
# The validation is done at each iteration based on three filters.
# The first filter is based on the min/max ranges. Observe that these values are defined in
# terms of minimum and maximum displacement in pixel/frames.
settings.MinMax_U_disp = (-10, 10)
settings.MinMax_V_disp = (-10, 10)
# The second filter is based on the global STD threshold
settings.std_threshold = 5  # threshold of the std validation
# The third filter is the median test (not normalized at the moment)
settings.median_threshold = 3  # threshold of the median validation
# On the last iteration, an additional validation can be done based on the S/N.
settings.median_size = 1 #defines the size of the local median
'Validation based on the signal to noise ratio'
# Note: only available when extract_sig2noise==True and only for the last
# pass of the interrogation
# Enable the signal to noise ratio validation. Options: True or False
settings.do_sig2noise_validation = True # This is time consuming
# minmum signal to noise ratio that is need for a valid vector
settings.sig2noise_threshold = 1.2
'Outlier replacement or Smoothing options'
# Replacment options for vectors which are masked as invalid by the validation
settings.replace_vectors = True # Enable the replacment. Chosse: True or False
settings.smoothn=True #Enables smoothing of the displacemenet field
settings.smoothn_p=0.5 # This is a smoothing parameter
# select a method to replace the outliers: 'localmean', 'disk', 'distance'
settings.filter_method = 'localmean'
# maximum iterations performed to replace the outliers
settings.max_filter_iteration = 4
settings.filter_kernel_size = 2  # kernel size for the localmean method
'Output options'
# Select if you want to save the plotted vectorfield: True or False
settings.save_plot = True
# Choose wether you want to see the vectorfield or not :True or False
settings.show_plot = False
settings.scale_plot = 200 # select a value to scale the quiver plot of the vectorfield
# run the script with the given settings
settings.counter = 0

picture_amount = 20
value_array = ['circular', 'linear']
for j in range(0, len(value_array)):
    settings.correlation_method = value_array[j]
    i = 1
    for i in range(1, picture_amount+1):
        settings.frame_pattern_a = 'Cyl_R_150_k7_%03da.tif' % i
        settings.frame_pattern_b = 'Cyl_R_150_k7_%03db.tif' % i
        settings.counter = i
        PIV_windef(settings)
    
    import os  # This is to understand which separator in the paths (/ or \)
    
    import matplotlib.pyplot as plt  # This is to plot things
    import numpy as np  # This is for doing math
    
    ################## Post Processing of the PIV Field.
    
    ## Step 1: Read all the files and (optional) make a video out of it.
    
    FOLDER = 'Results_PIV' + os.sep + 'Open_PIV_results_Test_1'
    n_t = picture_amount  # number of steps.
    
    # Read file number 10 (Check the string construction)
    Name = FOLDER + os.sep + 'field_A%03d' % 1 + '.txt'  # Check it out: print(Name)
    # Read data from a file
    DATA = np.genfromtxt(Name)  # Here we have the four colums
    nxny = DATA.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    n_s = 2 * nxny
    ## 1. Reconstruct Mesh from file
    X_S = DATA[:, 0]
    Y_S = DATA[:, 1]
    # Number of n_X/n_Y from forward differences
    GRAD_Y = np.diff(Y_S)
    # Depending on the reshaping performed, one of the two will start with
    # non-zero gradient. The other will have zero gradient only on the change.
    IND_X = np.where(GRAD_Y != 0)
    DAT = IND_X[0]
    n_y = DAT[0] + 1
    # Reshaping the grid from the data
    n_x = (nxny // (n_y))  # Carefull with integer and float!
    Xg = (X_S.reshape((n_x, n_y)))
    Yg = (Y_S.reshape((n_x, n_y)))  # This is now the mesh! 60x114.
    # Reshape also the velocity components
    V_X = DATA[:, 2]  # U component
    V_Y = DATA[:, 3]  # V component
    # Put both components as fields in the grid
    Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
    Vxg = (V_X.reshape((n_x, n_y)))
    Vyg = (V_Y.reshape((n_x, n_y)))
    Magn = (Mod.reshape((n_x, n_y)))
    
    fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
    # Plot Contours and quiver
    plt.contourf(Xg * 1000, Yg * 1000, Magn)
    plt.quiver(X_S * 1000, Y_S * 1000, V_X, V_Y)
    
    ###### Step 2: Compute the Mean Flow and the standard deviation.
    # The mean flow can be computed by assembling first the DATA matrices D_U and D_V
    D_U = np.zeros((n_s, n_t))
    D_V = np.zeros((n_s, n_t))
    # Loop over all the files: we make a giff and create the Data Matrices
    GIFNAME = 'Giff_Velocity.gif'
    Fol_Out = 'Gif_Images'
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)
    images = []
    
    D_U = np.zeros((n_x * n_y, n_t))  # Initialize the Data matrix for U Field.
    D_V = np.zeros((n_x * n_y, n_t))  # Initialize the Data matrix for V Field.
    
    for k in range(0, n_t):
        # Read file number 10 (Check the string construction)
        Name = FOLDER + os.sep + 'field_A%03d' % (k+1) + '.txt'  # Check it out: print(Name)
        # We prepare the new name for the image to export
        NameOUT = Fol_Out + os.sep + 'Im%03d' % (k+1) + '.png'  # Check it out: print(Name)
        # Read data from a file
        DATA = np.genfromtxt(Name)  # Here we have the four colums
        V_X = DATA[:, 2]  # U component
        V_Y = DATA[:, 3]  # V component
        # Put both components as fields in the grid
        Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
        Vxg = (V_X.reshape((n_x, n_y)))
        Vyg = (V_Y.reshape((n_x, n_y)))
        Magn = (Mod.reshape((n_x, n_y)))
        # Prepare the D_MATRIX
        D_U[:, k] = V_X
        D_V[:, k] = V_Y
        # Open the figure
        fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
        # Or you can plot it as streamlines
        #plt.contourf(Xg *1000 , Yg*1000 , Magn)
        # One possibility is to use quiver
        STEPx = 1
        STEPy = 1
        plt.quiver(Xg[::STEPx, ::STEPy] * 1000, Yg[::STEPx, ::STEPy] * 1000,
                   Vxg[::STEPx, ::STEPy], Vyg[::STEPx, ::STEPy], color='k')  # Create a quiver (arrows) plot
        plt.rc('text', usetex=True)  # This is Miguel's customization
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
    
        # ax is an object, which we could get also using ax=plt.gca() 
        # We can now modify all the properties of this obect 
        # In this exercise we follow an object-oriented approach. These
        # are all the properties we modify
        ax.set_aspect('equal')  # Set equal aspect ratio
        ax.set_xlabel('$x[mm]$', fontsize=18)
        ax.set_ylabel('$y[mm]$', fontsize=18)
        #   ax.set_title('Velocity Field via TR-PIV',fontsize=18)
        ax.set_xticks(np.arange(0, 40, 10))
        ax.set_yticks(np.arange(5, 30, 5))
        #   ax.set_xlim([0,43])
        #   ax.set_ylim(0,28)
        #   ax.invert_yaxis() # Invert Axis for plotting purpose
        # Observe that the order at which you run these commands is important!
        # Important: we fix the same c axis for every image (avoid flickering)
        plt.clim(0, 10)
        plt.colorbar()  # We show the colorbar
        plt.savefig(NameOUT, dpi=100)
        plt.close(fig)
        print('Image ' + str(k+1) + ' of ' + str(n_t))
    
    ########################################################################
    ## Compute the mean flow and show it
    ########################################################################
    
    D_MEAN_U = np.mean(D_U, axis=1)  # Mean of the u's
    D_MEAN_V = np.mean(D_V, axis=1)  # Mean of the v's
    Mod = np.sqrt(D_MEAN_U ** 2 + D_MEAN_V ** 2)  # Modulus of the mean
    
    Vxg = (D_MEAN_U.reshape((n_x, n_y)))
    Vyg = (D_MEAN_V.reshape((n_x, n_y)))
    Magn = (Mod.reshape((n_x, n_y)))
    
    fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
    # Or you can plot it as streamlines
    Magn = Magn / np.max(Magn)
    plt.contourf(Xg * 1000, Yg * 1000, Magn)
    # One possibility is to use quiver
    STEPx = 1
    STEPy = 1
    plt.quiver(Xg[::STEPx, ::STEPy] * 1000, Yg[::STEPx, ::STEPy] * 1000,
               Vxg[::STEPx, ::STEPy], Vyg[::STEPx, ::STEPy], color='k')  # Create a quiver (arrows) plot
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.set_xlabel('$x[pixels]$', fontsize=18)
    ax.set_ylabel('$y[pixels]$', fontsize=18)
    # ax.set_title('Velocity Field via TR-PIV',fontsize=18)
    # ax.set_xticks(np.arange(0,40,10))
    # ax.set_yticks(np.arange(5,30,5))
    # ax.invert_yaxis() # Invert Axis for plotting purpose
    # Observe that the order at which you run these commands is important!
    # Important: we fix the same c axis for every image (avoid flickering)
    plt.clim(0, 10)
    plt.colorbar()  # We show the colorbar
    plt.savefig('RESULTS' + os.sep + 'MEAN_FLOW_from_data_' + settings.correlation_method + '.png', dpi=100)
    plt.close(fig)
    
    """
    Here the calculation of the theoretical velocity field starts
    The velocities are calculated in polar coordinates and then transformed back into cartesian coordinates
    with the same grid as the PIV has
    """
    
    radius = 150.0
    u_infty = 1
    
    #define the radial velocity
    def radial_velocity(rho, theta):
        return (u_infty * (1 - 3 * radius / (rho * 2.0) + radius**3 / (2.0 * rho**3)) * np.cos(theta)) 
    
    #define the angular velocity
    def angular_velocity(rho, theta):
        return (u_infty * (1 - 3 * radius / (4.0*rho) - radius**3 / (4 * rho**3)) * np.sin(theta))
    
    #define the radius calculation
    def calc_rho(x):
        return np.sqrt(x[0]**2+x[1]**2)
    
    #define the angle calculation
    def calc_angle(x):
        return np.arctan2(x[1], x[0])
    
    #create a 2d array with the centre of each interrogation window as the coordinates
    #important to think about is that the origin is in the middle of the cylinder
    cartesian_grid = np.zeros((2, 30, 30))
    for i in range(-15, 15):
        cartesian_grid[0, :, i+15] = 32 * i +12
    for j in range(-15, 15):
        cartesian_grid[1, j+15, :] =  32 * j +12 
    #flip the y components to get the correct orientation
    cartesian_grid = cartesian_grid[:, ::-1,:]
    
    #calculate the radial and angular component of the vector
    rho = calc_rho(cartesian_grid[:,:,:])
    angle = calc_angle(cartesian_grid[:,:,:])
    
    #calculate the velocities in polar coordinates
    ang_vel = angular_velocity(rho, angle)
    rad_vel = radial_velocity(rho, angle)
    
    #calculate the velocities in cartesian coordinates
    v_x_theoretical = np.cos(angle) * radial_velocity(rho, angle) +  np.sin(angle) * angular_velocity(rho, angle)
    v_y_theoretical = np.sin(angle) * radial_velocity(rho, angle) -  np.cos(angle) * angular_velocity(rho, angle)
    mag_theoretical = np.sqrt(v_x_theoretical**2 + v_y_theoretical**2)
    
    ######################################################################
    ##extract the velocity profile shortly befor the cylinder#############
    ######################################################################
    
    mag_plot_theoretical = mag_theoretical[:, 5]
    x_val = np.arange(32, 992, 32) #corresponding y values for plotting
    mag_plot_theoretical = mag_plot_theoretical / np.max(mag_plot_theoretical)
    
    plt.plot(x_val, mag_plot_theoretical)
    plt.close('all')
    
    ## Step 3: Extract three velocity profile. Plot them in self similar forms
    # We will store the profiles in a matrix
    Prof_U_data = Magn[:, 5]
    Prof_U_data = Prof_U_data / np.max(Prof_U_data)
    
    #calculate the Mean Root Square 
    a = mag_plot_theoretical - Prof_U_data
    b = a**2
    mrs = sum(b)
    
    fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
    plt.plot(x_val, mag_plot_theoretical, 'ko', label = 'Theoretical data')
    plt.plot(x_val, Prof_U_data, 'rs', label = 'Experimental data')
    #plt.plot(y_e, Prof_U[:, 1], 'rs', label='$x_2=20mm$')
    #plt.plot(y_e, Prof_U[:, 2], 'bv', label='$x_3=30mm$')
    plt.title('Root Mean Square = %.5f' % mrs)
    ax.set_xlabel('$y[pixels]$', fontsize=18)
    ax.set_ylabel('$\hat{U}$', fontsize=18)
    plt.legend()
    plt.savefig('RESULTS' + os.sep + 'Velocity_comparison_' + settings.correlation_method + '.png', dpi=100)
    plt.show()
    plt.close(fig)
#    ax.set_xlim([0, 1000])
#    ax.set_ylim([0, 1])
