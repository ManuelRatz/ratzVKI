# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:40:18 2019

@author: mendez
"""

import os  # This is to understand which separator in the paths (/ or \)

import matplotlib.pyplot as plt  # This is to plot things
import numpy as np  # This is for doing math

################## Post Processing of the PIV Field.

## Step 1: Read all the files and (optional) make a video out of it.

FOLDER = 'Results_PIV' + os.sep + 'Open_PIV_results_Test_1'
n_t = 20  # number of steps.

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
    print('Image n ' + str(k) + ' of ' + str(n_t))


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
ax.set_xlabel('$x[mm]$', fontsize=18)
ax.set_ylabel('$y[mm]$', fontsize=18)
# ax.set_title('Velocity Field via TR-PIV',fontsize=18)
# ax.set_xticks(np.arange(0,40,10))
# ax.set_yticks(np.arange(5,30,5))
#   ax.set_xlim([0,43])
#   ax.set_ylim(0,28)
#   ax.invert_yaxis() # Invert Axis for plotting purpose
# Observe that the order at which you run these commands is important!
# Important: we fix the same c axis for every image (avoid flickering)
plt.clim(0, 10)
plt.colorbar()  # We show the colorbar
plt.savefig('MEAN_FLOW_from_data', dpi=100)
plt.close(fig)


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
cartesian_grid = np.zeros((2, 61, 61))
for i in range(-30, 31):
    cartesian_grid[0, :, i+30] = 16 * i -4
for j in range(-30, 31):
    cartesian_grid[1, j+30, :] =  16 * j + -4 
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


#plot the velocity field
#plt.quiver(cartesian_grid[0,:,:], cartesian_grid[1,:,:], v_x, v_y)

######################################################################
##extract the velocity profile shortly befor the cylinder#############
######################################################################

mag_plot_theoretical = mag_theoretical[:, 2]
x_val = np.arange(16, 992, 16) #corresponding y values for plotting
mag_plot_theoretical = mag_plot_theoretical / np.max(mag_plot_theoretical)

plt.plot(x_val, mag_plot_theoretical)
plt.close('all')

## Step 3: Extract three velocity profile. Plot them in self similar forms
# We will store the profiles in a matrix
Prof_U_data = Magn[:, 2]
Prof_U_data = Prof_U_data / np.max(Prof_U_data)
# The y axis, for plotting purposes is
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1])
plt.plot(x_val, mag_plot_theoretical, 'ko', label = 'Theoretical data')
#plt.plot(x_val, Prof_U_data, 'rs', label = 'Experimental data')
plt.legend()
plt.savefig('Comparison of velocity profiles', dpi = 100)


"""
YC = 21.6  # Assume the centerline is approximately at 21.6 mm.
# Obs: for the moment this values is just a quick guess: can you think of a better way to do this?
y_e = Y_axis * 1000 - YC  # This is the experimental grid in mm


fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
plt.plot(y_e, Prof_U[:, 0], 'ko', label='Experimental data')
ax.set_xlabel('$x[mm]$', fontsize=18)
ax.set_ylabel('$|V_x|$', fontsize=18)
ax.set_title('Velocity Profiles', fontsize=18)
plt.legend()
plt.savefig('Vel_Profiles.png', dpi=100)
plt.show()
plt.close(fig)
"""