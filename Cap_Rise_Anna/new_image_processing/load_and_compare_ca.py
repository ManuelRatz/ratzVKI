# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:14:42 2021

@author: Manuel Ratz
"""

import sys
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\PIV_Campaign_Processing')
sys.path.append('C:\\Users\manue\Documents\GitHub\\ratzVKI\Cap_Rise_Anna\new_processing')

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate           # for setting up interpolation functions
from scipy.integrate import odeint      # for solving the ode
import os
import scipy.signal as sci 

mu_w = 0.000948
rho_w = 997.770
sigma_w = 0.0724

Fol_In = 'C:\Anna\Rise\Water\P1500_C30\A'
Case = 'P1500_C30_A'

def load_txt_files(Fol_In, Case):
    """
    Function to load the result files from the LeDaR interface detection. All
    files are already shifted to start at Frame0

    Parameters
    ----------
    Fol_In : str
        Location to the filepath were each case is stored.
    Case : str
        The given test case to plot.

    Returns
    -------
    pressure : 1d np.array
        Pressure in the facility in pa.
    h_avg : 1d np.array
        Average height of the interface in milimeter.
    ca_gauss : 1d np.array
        Contact angle from the gaussian fitting in degrees.
    ca_cosh : 1d np.array
        Contact angle from the hyperbolic cosine fit in degrees.
    fit_gauss : 1d np.array
        Average curvature of the gaussian fitting in 1/mm.
    fit_exp : 1d np.array
        Average curvature of the hyperbolic cosine fit in 1/mm.

    """
    # navigate to the data files
    Folder = os.path.join(Fol_In, 'data_files')
    # load each file, convert the angles to radians
    pressure = np.genfromtxt(os.path.join(Folder, Case + '_pressure.txt'))
    h_avg = np.genfromtxt(os.path.join(Folder, Case + '_h_avg.txt'))
    ca_gauss = np.genfromtxt(os.path.join(Folder, Case + '_ca_gauss.txt'))/180*np.pi
    ca_cosh = np.genfromtxt(os.path.join(Folder, Case + '_ca_cosh.txt'))/180*np.pi
    fit_gauss = np.genfromtxt(os.path.join(Folder, Case + '_gauss_curvature.txt'))
    fit_exp = np.genfromtxt(os.path.join(Folder, Case + '_cosh_curvature.txt'))
    # return all the files
    return pressure, h_avg, ca_gauss, ca_cosh, fit_gauss, fit_exp

def solve_equation(pressure_call, height_call, method = 'Angle', ca_call = None,\
                   curv_call = None, pressure_loss = 0):
    """
    Function to solve the governing ODE after giving each signal

    Parameters
    ----------
    pressure_call : 1d np.array
        Pressure signal used in the equation.
    height_call : 1d np.array
        Average height of the interface used in the equation.
    method : str, optional
        How to calculate the laplace pressure, can either be *Angle* or
        *Curvature*. The default is 'Angle'.
    ca_call : 1d np.array, optional
        Signal of the contact angle in radians. The default is None.
    curv_call : 1d np.array, optional
        Signal of the curvature in 1/mm. The default is None.

    Raises
    ------
    ValueError
        If the wrong case is given or no contact angle/curvature is given.

    Returns
    -------
    solution : 2d np.array
        2 columns containing the velocity and position of the solution in m/s and m.

    """
    # set the constants (this is for water)
    g      = 9.81;          # gravity (m/s2)
    r      = 0.0025         # radius of the tube (m)
    delta  = 2* r           # channel width (m)
    l      = 100*r          # depth of the channel
    # set up the time array
    t = np.linspace(0, 4, 2000, endpoint = True)
    pres_interp = interpolate.splrep(t, pressure_call)
    def pres(t):
        return interpolate.splev(t, pres_interp)
    if ca_call is not None:
        ca_interp = interpolate.splrep(t, ca_call)
        def ca(t):
            return interpolate.splev(t, ca_interp)
    elif curv_call is not None:
        curv_interp = interpolate.splrep(t, curv_call)
        def curv(t):
            return interpolate.splev(t, curv_interp)
    else:
        raise ValueError('Either a contact angle or a curvature must be given to the solver')
        
    def ode_ca(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu_w*U*Y/(rho_w*l*delta**2) + pres(t)/rho_w\
                          + 2.0*(l+delta)*sigma_w*np.cos(ca(t))/(rho_w*l*delta) - (1+pressure_loss)*U*abs(U))
        dydt = U
        return [dudt, dydt]
    def ode_curv(X, t):
        U, Y = X
        dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu_w*U*Y/(rho_w*l*delta**2) + pres(t)/rho_w\
                          + l*sigma_w*curv(t)/(rho_w*l*delta) - (1+pressure_loss)*U*abs(U))
        dydt = U
        return [dudt, dydt]
    h0 = height_call[0]/1000
    velocity = np.gradient(height_call/1000)
    dh0 = velocity[0]/(0.002)
    X0 = np.array([dh0, h0])
    if method == 'Angle':    
        solution = odeint(ode_ca, X0, t)*1000
    elif method == 'Curvature':    
        solution = odeint(ode_curv, X0, t)*1000
    else:
        raise ValueError('Wrong method, must be *Angle* or *Curvature*')
    return solution
def filter_signal(signal, cutoff_frequency):
    left_attach = 2*signal[0]-signal[::-1]
    right_attach = 2*signal[-1]-signal[::-1]
    continued_signal = np.hstack((left_attach, signal, right_attach))

    windows = sci.firwin(numtaps = signal.shape[0]//10, cutoff = cutoff_frequency,\
                         window='hamming', fs = 500)
    example_filtered = sci.filtfilt(b = windows, a = [1], x = continued_signal)
    
    clip_index = signal.shape[0]
    clipped_signal = example_filtered[clip_index:2*clip_index]
    return clipped_signal
#%%
pressure, h_avg, ca_gauss, ca_cosh, curv_gauss, curv_cosh = load_txt_files(Fol_In, Case)

pressure_filtered = filter_signal(signal = pressure, cutoff_frequency = 8)
ca_gauss_filtered = filter_signal(signal = ca_gauss, cutoff_frequency = 10)
ca_cosh_filtered = filter_signal(signal = ca_cosh, cutoff_frequency = 10)
curv_gauss_filtered = filter_signal(signal = curv_gauss, cutoff_frequency = 10)
curv_cosh_filtered = filter_signal(signal = curv_cosh, cutoff_frequency = 10)
h_avg_filtered = filter_signal(signal = h_avg, cutoff_frequency = 10)


""" Solutions with the different contact angles """
solution_gauss_filt_ca = solve_equation(pressure_call = pressure, height_call = h_avg, pressure_loss = 0,\
                                     method = 'Angle', ca_call = ca_gauss_filtered)[:,1]
solution_gauss_filt_ca_pres_loss = solve_equation(pressure_call = pressure, height_call = h_avg, pressure_loss = 1/3,\
                                     method = 'Angle', ca_call = ca_gauss_filtered)[:,1]
# solution_gauss_unfilt_ca = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Angle', ca_call = ca_gauss)[:,1]
# solution_cosh_filt_ca = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Angle', ca_call = ca_cosh_filtered)[:,1]
# solution_cosh_unfilt_ca = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Angle', ca_call = ca_cosh)[:,1]

# """ Solutions with the different curvatures """
# solution_gauss_filt_curv = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Curvature', curv_call = curv_gauss_filtered)[:,1]
# solution_gauss_unfilt_curv = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Curvature', curv_call = curv_gauss)[:,1]
# solution_cosh_filt_curv = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Curvature', curv_call = curv_cosh_filtered)[:,1]
# solution_cosh_unfilt_curv = solve_equation(pressure_call = pressure, height_call = h_avg,\
#                                      method = 'Curvature', curv_call = curv_cosh)[:,1]
#%%
plt.figure()
plt.plot(solution_gauss_filt_ca)
plt.plot(solution_gauss_filt_ca_pres_loss)

#%%
"""Plots for the Miguel Presentation"""
# plt.figure()
# plt.plot(solution_gauss_filt_ca, label = 'Gauss')
# plt.plot(solution_cosh_filt_ca, label = 'Cosh')
# plt.legend()
# plt.title('Contact angle comparison')
# plt.savefig('ca_sol.png', dpi = 400)

# plt.figure()
# plt.plot(solution_gauss_filt_curv, label = 'Gauss')
# plt.plot(solution_cosh_filt_curv, label = 'Cosh')
# plt.legend()
# plt.title('Curvature comparison')
# plt.savefig('curv_sol.png', dpi = 400)

# plt.figure()
# plt.plot(solution_gauss_filt_ca, label = 'Contact Angle')
# plt.plot(solution_gauss_filt_curv, label = 'Curvature')
# plt.legend()
# plt.title('Gauss comparison')
# plt.savefig('gauss_sol.png', dpi = 400)

# plt.figure()
# plt.plot(solution_cosh_filt_ca, label = 'Contact Angle')
# plt.plot(solution_cosh_filt_curv, label = 'Curvature')
# plt.legend()
# plt.title('Cosh comparison')
# plt.savefig('cosh_sol.png', dpi = 400)

# plt.figure()
# plt.plot(solution_cosh_filt_ca, label = 'Cosh, CA', lw = 0.75)
# plt.plot(solution_cosh_filt_curv, label = 'Cosh Curv', lw = 0.75)
# plt.plot(solution_gauss_filt_ca, label = 'Gauss, CA', lw = 0.75)
# plt.plot(solution_gauss_filt_curv, label = 'Gauss Curv', lw = 0.75)
# # plt.ylim(99, 108)
# # plt.xlim(1000, 1750)
# plt.title('Solution global all')
# plt.legend()
# plt.savefig('all_sol.png', dpi = 400)

# plt.figure()
# plt.plot(2*np.cos(ca_gauss_filtered), label = 'Contact angle')
# plt.plot(curv_gauss_filtered, label = 'Curvature')
# plt.legend()
# plt.title('Laplace term gauss')
# plt.savefig('gauss_lap.png', dpi = 400)

# plt.figure()
# plt.plot(2*np.cos(ca_cosh_filtered), label = 'Contact angle')
# plt.plot(curv_cosh_filtered, label = 'Curvature')
# plt.legend()
# plt.title('Laplace term cosh')
# plt.savefig('cosh_lap.png', dpi = 400)

# plt.figure()
# plt.plot(2*np.cos(ca_gauss_filtered), label = 'Ca, Gauss', lw = 0.75)
# plt.plot(curv_gauss_filtered, label = 'Curv, Gauss', lw = 0.75)
# plt.plot(2*np.cos(ca_cosh_filtered), label = 'Ca, Gauss', lw = 0.75)
# plt.plot(curv_cosh_filtered, label = 'Curv, Cosh', lw = 0.75)
# plt.legend()
# plt.title('Laplace term all')
# plt.savefig('all_lap.png', dpi = 400)

# plt.figure()
# plt.plot(curv_gauss_filtered, label = 'Gauss')
# plt.plot(curv_cosh_filtered, label = 'Cosh')
# plt.legend()
# plt.title('Curvature Comparison')
# plt.savefig('curv.png', dpi = 400)

# plt.figure()
# plt.plot(ca_gauss_filtered*180/np.pi, label = 'Gauss')
# plt.plot(ca_cosh_filtered*180/np.pi, label = 'Cosh')
# plt.legend()
# plt.title('Contact angle comparison')
# plt.savefig('ca.png', dpi = 400)

# # plt.figure()
# # plt.plot(ca_gauss)
# # plt.plot(ca_gauss_filtered)

# # plt.figure()
# # plt.plot(ca_cosh)
# # plt.plot(ca_cosh_filtered)

# # plt.figure()
# # plt.plot(curv_gauss)
# # plt.plot(curv_gauss_filtered)

# # plt.figure()
# # plt.plot(curv_cosh)
# # plt.plot(curv_cosh_filtered)

# # plt.figure()
# # plt.plot(pressure)
# # plt.plot(pressure_filtered)

