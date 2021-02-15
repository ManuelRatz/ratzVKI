# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:28:15 2021

@author: mendez
"""

import numpy as np
# Some interpolators

def interp_Space_1D(phi,x_old,x_new,n_B,alpha,l,P):
     # Phi is the spatial basis (old grid)
     # x_old is the old grid
     # x_new is the (given) new grid
     # n_B is the number of RBFs
     # alpha is the penalization parameter    
    return phi_new


def interp_Space_2D(phi,x_old,x_new,n_B,alpha,l1,l2,P):
     # Phi is the spatial basis (old grid)
     # x_old is the old grid
     # x_new is the (given) new grid
     # n_B is the number of RBFs
     # alpha is the penalization parameter   
     # P is the kind of penalty: 
        # 1 is L1 penalty on the weights
        # 2 is L2 penalty on the weights
        
        
    return phi_new

def interp_Time(psi_old,t_old,t_new,n_B,sigma,P,alpha):
     # Phi is the spatial basis (old grid)
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
     T_P=np.linspace(t_old.min(),t_old.max(),n_B)
     n_t_o=t_old.shape[0]
     Psi_T_o=np.zeros((n_t_o,n_B)) # Initialize Data Matrix    
     for j in range(0,n_B):
       Psi_T_o[:,j]=np.exp(-(T_P[j]-t_old)**2/sigma**2)  
    # Solve the regularized regression (for now only L2)
    # min J(w)= ||psi_old-Psi_T_o*w_psi||_2+alpha*||w_psi||_2
     w_psi=np.linalg.inv(Psi_T_o.T.dot(Psi_T_o)+\
          alpha*np.eye(n_B)).dot(Psi_T_o.T).dot(psi_S)    
    # Now we prepare the matrix for interpolation 
     n_x_n=t_new.shape[0]
     Psi_T_n=np.zeros((n_x_n,n_B)) # Initialize Data Matrix    
     for j in range(0,n_B):
       Psi_T_n[:,j]=np.exp(-(T_P[j]-t_new)**2/sigma**2) 
    # Finally perform the interpolation
     phi_new=Psi_T_n.dot(w_psi)+Y_MEAN    
    
     return phi_new


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





