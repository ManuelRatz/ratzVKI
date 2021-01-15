# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:07:36 2020

@author: Manuel Ratz
@description : Script to test the smoothing on 1 velocity profile for a fall.
    Loads a 3d tensor of a fall that has been smoothed in time and along the
    vertical axis and smoothes it along the horizontal one using a sine
    transformation
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')

import matplotlib.pyplot as plt
import numpy as np

""" Here we load the unsmoothe velocity fields and prepare them for the POD
    by extending the velocity profile with the velocity of the top row once
    the interface comes into the FOV """

# load the data
profile_v_load = np.load('v_tensor.npy') # vertical velocity
profile_u_load = np.load('u_tensor.npy') # horizontal velocity
x_old = np.load('x_coordinates.npy') # x coordinates (1d)

# create the y spacing (we have 103 vertical points and a width of 24 with 12 overlap)
y_old = np.linspace(24, 12*103+24, 104)[::-1]

# initialize the padded profiles
profile_v = np.zeros(profile_v_load.shape)
profile_u = np.zeros(profile_u_load.shape)
# loop over the timesteps
for i in range(0, profile_v.shape[0]):
    # copy the profiles into a temporary storage
    prof_v = profile_v_load[i,:,:]
    prof_u = profile_u_load[i,:,:]
    # check for the interface (above it, all values are 1000)
    valid_idx = np.argmax(prof_v[:,9] < 800)
    # create a dummy for the extended profiles
    Dummy_v = np.ones((valid_idx, prof_v.shape[1]))*prof_v[valid_idx,:]
    Dummy_u = np.ones((valid_idx, prof_u.shape[1]))*prof_u[valid_idx,:]
    # stack them on the valid points
    profile_v[i,:,:] = np.vstack((Dummy_v, prof_v[valid_idx:]))
    profile_u[i,:,:] = np.vstack((Dummy_u, prof_u[valid_idx:]))

""" This is the step of the POD. We reshape everything into column vectors, do
    the POD and then calculate the new velocity field with removed outliers """

# initialize the velocity tensors for the POD (of size n_x*n_y, n_t)
U_POD_load = np.zeros((profile_u.shape[1]*profile_u.shape[2], profile_u.shape[0]))
V_POD_load = np.zeros((profile_v.shape[1]*profile_v.shape[2], profile_v.shape[0]))
# reshape them into column arrays
for k in range(0,325):
    V_POD_load[:,k] = np.reshape(profile_v[k,:,:],((104*23,)))
    U_POD_load[:,k] = np.reshape(profile_u[k,:,:],((104*23,)))
    
# import POD functions
from functions_Miguel import POD, interp_Time, qfield, high_pass

# number of POD modes
R_u = 25
R_v = 35
# do the POD
Phi_u, Sigma_u, Psi_u = POD(U_POD_load, R_u)
Phi_v, Sigma_v, Psi_v = POD(V_POD_load, R_v)
    

# recalculate the velocity fields
V_POD=np.linalg.multi_dot([Phi_v,np.diag(Sigma_v),np.transpose(Psi_v)]) 
U_POD=np.linalg.multi_dot([Phi_u,np.diag(Sigma_u),np.transpose(Psi_u)])

# U_pod_old = U_POD[:,time_step].reshape(104, 23)[top_idx:,1:-1]
# V_pod_old = V_POD[:,time_step].reshape(104, 23)[top_idx:,1:-1]

# upod = U_pod_old
# vpod = V_pod_old

# test_profile = vpod[10,:]
# # extract the original velocity profile and grid

# test_filt = interp_Time(test_profile, x_old, x_new, n_B = 15, sigma = 15, P=2, alpha = 0.1)
# plt.figure()
# plt.plot(X_grid_old[10,:], v_old[7,:]) 
# plt.plot(X_grid_old[10,:], vpod[7,:], color = 'r') 
# plt.plot(x_new, test_filt, color = 'k')
 
# estimate the errors
Error=np.linalg.norm(U_POD-U_POD_load)/np.linalg.norm(U_POD_load)
print('Convergence Error U: E_C='+"{:.2f}".format(Error*100)+' %') 
Error=np.linalg.norm(V_POD-V_POD_load)/np.linalg.norm(V_POD_load)
print('Convergence Error V: E_C='+"{:.2f}".format(Error*100)+' %')

""" We do an additional smoothing in time with RBFs. Interpolation in time
    is possible but not what we are after """

# time discretization
dt_1=1/1200; n_t1=325
# set up the time arrays, n is the interpolation, if set to 1 there is none
n = 1
t_1=dt_1*np.linspace(0,n_t1-1,n_t1);T=t_1.max()
t_2=np.linspace(0,T,n_t1*n) 

# number of radial basis functions along the time axis
n_B = 500
# initialize the interpolated Psi-s
Psi_u_interp = np.zeros((t_2.shape[0], R_u))
Psi_v_interp = np.zeros((t_2.shape[0], R_v))
# loop over all the POD modes
for i in range(0, R_u):
    Psi_u_interp[:,i]=interp_Time(Psi_u[:,i],t_1,t_2,n_B,0.01,2,alpha=0.1)
for i in range(0, R_v):
    Psi_v_interp[:,i]=interp_Time(Psi_v[:,i],t_1,t_2,n_B,0.01,2,alpha=0.1)

""" Here we do the interpolation and smoothing in space. For now this is not 
    being outsourced to the functions file, but will be soon """

# set up the new x grid, the multiplication in the end is the refinement
x_new = np.linspace(x_old.min(), x_old.max(), x_old.shape[0]*2)
y_new = np.linspace(y_old.min(), y_old.max(), y_old.shape[0]*2)

# get the number of grid points on the old and new grid
n_x_new = x_new.shape[0]
n_y_new = y_new.shape[0]
n_x_old = x_old.shape[0]
n_y_old = y_old.shape[0]


# define the two grids
X_old, Y_old = np.meshgrid(x_old, y_old)
X_new, Y_new = np.meshgrid(x_new, y_new)

# reshape the grids into column vectors
X_old = X_old.reshape(n_x_old*n_y_old)
Y_old = Y_old.reshape(n_x_old*n_y_old)
X_new = X_new.reshape(n_x_new*n_y_new)
Y_new = Y_new.reshape(n_x_new*n_y_new)

# number of radial basis functions along the x and y axis respectively
# WARNING: High values lead to a huge increase of the computational cost
n_Bx = 15
n_By = 60
# initialize the interpolated Phi-s
Phi_v_interp = np.zeros((y_new.shape[0]*x_new.shape[0], R_v))
Phi_u_interp = np.zeros((y_new.shape[0]*x_new.shape[0], R_u))

# sigmas for the width of the gaussians in each axis
sigma_x = 15
sigma_y = 10

# initialize the old and new Phi's for the vertical and horizontal component
Phi_v_T_o = np.zeros((n_x_old*n_y_old, n_Bx*n_By))
Phi_v_T_n = np.zeros((n_x_new*n_y_new, n_Bx*n_By))
Phi_u_T_o = np.zeros((n_x_old*n_y_old, n_Bx*n_By))
Phi_u_T_n = np.zeros((n_x_new*n_y_new, n_Bx*n_By))

#define the centers of the RBFs, create the grid and reshape it into columns
X_c = np.linspace(x_old.min(),x_old.max(),n_Bx)
Y_c = np.linspace(y_old.min(),y_old.max(),n_By)
X_c, Y_c = np.meshgrid(X_c, Y_c)
X_c = X_c.reshape(n_Bx*n_By)
Y_c = Y_c.reshape(n_Bx*n_By)

# penalty for the size of the w-s
alpha = 0.1

# set up the Gaussians by looping over every RBF grid poing
for j in range(0, n_Bx*n_By):
    Phi_v_T_o[:,j] = np.exp(-((X_c[j]-X_old)**2/(2*sigma_x**2)+\
                            (Y_c[j]-Y_old)**2/(2*sigma_y**2)))
    Phi_v_T_n[:,j] = np.exp(-((X_c[j]-X_new)**2/(2*sigma_x**2)+\
                            (Y_c[j]-Y_new)**2/(2*sigma_y**2)))
    Phi_u_T_o[:,j] = np.exp(-((X_c[j]-X_old)**2/(2*sigma_x**2)+\
                            (Y_c[j]-Y_old)**2/(2*sigma_y**2)))
    Phi_u_T_n[:,j] = np.exp(-((X_c[j]-X_new)**2/(2*sigma_x**2)+\
                            (Y_c[j]-Y_new)**2/(2*sigma_y**2)))
    
# loop over every POD mode
for j in range(0, R_u):
    # update the user
    print(j+1) 
    # u component
    phi_old = Phi_u[:,j]
    # subtract the mean
    phi_mean = phi_old.mean()
    phi_s = phi_old - phi_mean
    # min J(w)= ||psi_old-Psi_T_o*w_psi||_2+alpha*||w_psi||_2
    # solve this equation directly (for large n_By and n_Bx this is costly)
    w_phi = np.linalg.inv(Phi_u_T_o.T.dot(Phi_u_T_o)+
      alpha*np.eye(n_Bx*n_By)).dot(Phi_u_T_o.T).dot(phi_s)  
    Phi_u_interp[:,j] = Phi_u_T_n.dot(w_phi) + phi_mean  

for j in range(0, R_v):
    # update the user
    print(j+1) 
    # v component
    phi_old = Phi_v[:,j]
    # subtract the mean
    phi_mean = phi_old.mean()
    phi_s = phi_old - phi_mean
    # min J(w)= ||psi_old-Psi_T_o*w_psi||_2+alpha*||w_psi||_2
    # solve this equation directly (for large n_By and n_Bx this is costly)
    w_phi = np.linalg.inv(Phi_v_T_o.T.dot(Phi_v_T_o)+
      alpha*np.eye(n_Bx*n_By)).dot(Phi_v_T_o.T).dot(phi_s)  
    Phi_v_interp[:,j] = Phi_v_T_n.dot(w_phi) + phi_mean 

idx = 1
plt.plot(Psi_u[:,idx], label = 'original')
plt.plot(Psi_u_interp[:,idx], label = 'smooth')
plt.legend()

# calculate the newly smoothed and interpolated velocity field
New_v = np.linalg.multi_dot([Phi_v_interp,np.diag(Sigma_v),np.transpose(Psi_v_interp)])
New_u = np.linalg.multi_dot([Phi_u_interp,np.diag(Sigma_u),np.transpose(Psi_u_interp)])

# reshape it back into a tensor
reshaped_v = New_v.reshape((n_y_new, n_x_new, n_t1))
reshaped_u = New_u.reshape((n_y_new, n_x_new, n_t1))

#%%

""" Here we create the figures for the abstract. We always crop the 0 padding
    at the edges because it messes up the high pass filtering. We create the
    old q field and 2 quiver plots, on on the old grid and one on the new one """

# top index of the array to cut, for time step 201 this is 3
top_idx = 3
time_step = 201

X_grid_old = X_old.reshape(n_y_old, n_x_old)[top_idx:,1:-1]
Y_grid_old = Y_old.reshape(n_y_old, n_x_old)[top_idx:,1:-1]

# extract the interpolated velocity profile and grid
X_grid_new = X_new.reshape(n_y_new, n_x_new)
Y_grid_new = Y_new.reshape(n_y_new, n_x_new)[::-1]
X_grid_new = X_grid_new[2*top_idx:,1:-1]
Y_grid_new = Y_grid_new[2*top_idx:,1:-1]

v_new = reshaped_v[::-1,:,:]
u_new = reshaped_u[::-1,:,:]

v_new = v_new[2*top_idx:,1:-1,time_step]
u_new = u_new[2*top_idx:,1:-1,time_step]

# some plot settings
plt.rc('font', family='serif')          # serif as text font
plt.rc('text', usetex=True)             # enable latex
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=22)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

v_old = profile_v[time_step,top_idx:,1:-1]
u_old = profile_u[time_step,top_idx:,1:-1]

# take the high pass filtered original velocity (for now we use mean subtraction)
# u_old_hp, v_old_hp = high_pass(u_old, v_old, sigma_y = 10, sigma_x = 15, truncate = 2, padded=False)
subt_u = u_old.mean()
subt_v = v_old.mean()
u_old_hp = u_old-subt_u
v_old_hp = v_old-subt_v

# create the figure
fig, ax = plt.subplots(figsize = (8,5))
ax.quiver(X_grid_old, Y_grid_old, u_old_hp, v_old_hp, scale = 750, color = 'k',\
          scale_units = 'height', units = 'width', width = 0.002)
ax.set_xlabel('$x$[mm]')
ax.set_xticks(np.linspace(0, 264, 6))
ax.set_xticklabels(np.linspace(0, 5, 6, dtype = int))
ax.set_xlim(0, 264)
ax.set_ylabel('$y$[mm]')
ax.set_yticks(1269-np.arange(0,1269,55))
ax.set_yticklabels(np.arange(0, 24, 1)[::-1])
ax.set_ylim(1075, 1224)
fig.tight_layout()
fig.savefig('original.png', dpi = 400)

# take the high pass filtered interpolated velocity (for now we use mean subtraction)
# u_new_hp, v_new_hp = high_pass(u_new, v_new, sigma_y = 10, sigma_x = 15, truncate = 2, padded=False)

u_new_hp = u_new-subt_u
v_new_hp = v_new-subt_v

# create the figure
Stp=1
fig, ax = plt.subplots(figsize = (8,5))
ax.quiver(X_grid_new[::Stp,::Stp], Y_grid_new[::Stp,::Stp], u_new_hp[::Stp,::Stp],\
          v_new_hp[::Stp,::Stp], scale = 550, color = 'k', scale_units = 'height',\
        units = 'width', width = 0.002)
ax.set_xlabel('$x$[mm]')
ax.set_xticks(np.linspace(0, 275, 6))
ax.set_xticklabels(np.linspace(0, 5, 6, dtype = int))
ax.set_xlim(0, 275)
ax.set_ylabel('$y$[mm]')
ax.set_yticks(1269-np.arange(0,1269,55))
ax.set_yticklabels(np.arange(0, 24, 1, dtype = int)[::-1])
ax.set_ylim(1075, 1224)
fig.tight_layout()
fig.savefig('interpolated.png', dpi = 400)



# calculate the qfield
q_field = qfield(X_grid_old, Y_grid_old, u_old, v_old)
# create the figure
Stp_x = 1
Stp_y = 2
fig, ax = plt.subplots(figsize = (4.5, 10))
cont = ax.contourf(X_grid_old, Y_grid_old, q_field)            
clb = fig.colorbar(mappable = cont, fraction=0.185, pad=0.1, drawedges = True, alpha = 1) # get the colorbar
clb.set_label('Q Field [1/s$^2$]') # set the colorbar label
ax.quiver(X_grid_old[::Stp_y,::Stp_x], Y_grid_old[::Stp_y,::Stp_x], u_old[::Stp_y,::Stp_x],\
          v_old[::Stp_y,::Stp_x], scale = 12000, color = 'k',\
          scale_units = 'height', units = 'width', width = 0.002)
ax.set_xlabel('$x$[mm]')
ax.set_xticks(np.linspace(12, 252, 6))
ax.set_xticklabels(np.linspace(0, 5, 6, dtype = int))
ax.set_xlim(12, 252)
ax.set_ylabel('$y$[mm]')
ax.set_yticks(np.arange(24,1269,220))
ax.set_yticklabels(np.arange(0, 24, 4, dtype = int))
ax.set_aspect(1)
ax.set_ylim(24, 1269)
fig.tight_layout(pad = 1.3)
fig.savefig('qfield.png', dpi = 400)