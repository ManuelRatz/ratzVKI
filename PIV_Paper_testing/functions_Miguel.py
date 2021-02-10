# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:28:15 2021

@author: mendez
"""

import numpy as np
from smoothn import smoothn
from scipy.ndimage import gaussian_filter
# Some interpolators

def interp_Space_1D(phi,x_old,x_new,n_B,alpha,l,P):
     # Phi is the spatial basis (old grid)
     # x_old is the old grid
     # x_new is the (given) new grid
     # n_B is the number of RBFs
     # alpha is the penalization parameter    
    return phi_new



def interp_Space_2D(phi, XY_old, XY_new, n_Bx, n_By, alpha, P):
     # Phi is the spatial basis (old grid)
     # x_old is the old grid
     # x_new is the (given) new grid
     # n_B is the number of RBFs
     # alpha is the penalization parameter   
     # P is the kind of penalty: 
        # 1 is L1 penalty on the weights
        # 2 is L2 penalty on the weights
    """
    x_old is 2d
    define grid, old and new
    n_Bx
    n_By
    grid is uniform and rectangular
    
    phi.shape = nx*ny, nB_x*nB_y
    for i in range nx*ny:
        # gaussian can be elliptical
    reshape phi_old
    
    solve for w_phi
    """

    return phi_new

def interp_Time(psi_old, t_old, t_new, n_B, sigma, P, alpha):
     # Psi is the spatial basis (old grid)
     # x_old is the old grid
     # x_new is the (given) new grid
     # n_B is the number of RBFs
     # alpha is the penalization parameter    
     # P is the kind of penalty: 
         # 1 is L1 penalty on the weights
         # 2 is L2 penalty on the weights
    
    #%% Step 1: Given the X axis, we fix the RBF collocation
    Y_MEAN=psi_old.mean()
    psi_S=psi_old-Y_MEAN
    T_p=np.linspace(t_old.min(),t_old.max(),n_B)
    n_t_o=t_old.shape[0]
    n_t_n=t_new.shape[0] 
    Psi_T_o=np.zeros((n_t_o,n_B)) # Initialize Data Matrix 
    Psi_T_n=np.zeros((n_t_n,n_B)) # Initialize Data Matrix   
    for j in range(0,n_B):
      Psi_T_o[:,j]=np.exp(-(T_p[j]-t_old)**2/sigma**2)  
      Psi_T_n[:,j]=np.exp(-(T_p[j]-t_new)**2/sigma**2) 
    # Solve the regularized regression (for now only L2)
    # min J(w)= ||psi_old-Psi_T_o*w_psi||_2+alpha*||w_psi||_2
    w_psi=np.linalg.inv(Psi_T_o.T.dot(Psi_T_o)+\
         alpha*np.eye(n_B)).dot(Psi_T_o.T).dot(psi_S)             
    # Finally perform the interpolation
    psi_new=Psi_T_n.dot(w_psi)+Y_MEAN    
   
    return psi_new


def POD(D,R):

    K=np.dot(D.T,D)
    Psi_P,Lambda_P,_=np.linalg.svd(K)
    # The POD has the unique feature of providing the amplitude of the modes
    # with no need of projection. The amplitudes are:
    Sigma_P=(Lambda_P)**0.5; 
    # We take R modes
    Sigma_P_t=Sigma_P[0:R]
    Sigma_P_Inv_V=1/Sigma_P_t
    # Accordingly we reduce psi_P
    Psi_P_t=Psi_P[:,0:R]
    # So we have the inverse
    Sigma_P_Inv=np.diag(Sigma_P_Inv_V)
    Phi=np.linalg.multi_dot([D,Psi_P[:,0:R],Sigma_P_Inv])
    
    return Phi, Sigma_P_t, Psi_P_t


def high_pass(u, v, sigma_y, sigma_x, truncate, padded = False):
    if padded == True:
        u = u[:,1:-1]
        v = v[:,1:-1]
    # get the blurred velocity field
    u_blur = gaussian_filter(u, sigma = sigma_x, mode = 'nearest', truncate = truncate)
    v_blur = gaussian_filter(v, sigma = sigma_y, mode = 'nearest', truncate = truncate)
    # subtract to get the high pass filtered velocity field
    u_filt = u - u_blur
    v_filt = v - v_blur
    # return the result
    return u_filt, v_filt

def qfield(x, y, u, v):
    # copy the arrays into dummys
    u_copy = np.copy(u)
    v_copy = np.copy(v)
    # smooth the velocities heavily
    u_smo, dum, dum, dum = smoothn(u_copy, s = 7.5)
    v_smo, dum, dum, dum = smoothn(v_copy, s = 7.5)
    # calculate the derivatives
    ux = np.gradient(u_smo, x[-1,:], axis = 1)
    uy = np.gradient(u_smo, y[:,0], axis = 0)
    vx = np.gradient(v_smo, x[-1,:], axis = 1)
    vy = np.gradient(v_smo, y[:,0], axis = 0)
    # calculate the qfield
    qfield = -0.5*(ux**2+2*vx*uy+vy**2)
    # return it
    return qfield
