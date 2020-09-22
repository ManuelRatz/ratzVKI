import numpy as np
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import odeint

#set the constants
g      = 9.81;         # Gravity (m/s2)
Patm   = 1e5  # atmospheric pressure, Pa
rhoL   = 1000;         #Fluid Density (kg/m3)
mu     = 8.90*10**(-4); # Fuid dynamic viscosity (Pa*s)
sigma  = 0.07197;      # surface tension (N/m)
delta  = 2* 0.0025   # channel width (m)
l      = 100      # depth of the channel
pressure_step = 990.89

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
FOLDER_OUT = 'own_equation_testing' + os.sep

#load the data
FOLDER = '..' + os.sep + 'experimental_data' + os.sep + '1500_pascal' + os.sep
lca_load = np.genfromtxt(FOLDER + 'LCA.txt')
rca_load = np.genfromtxt(FOLDER + 'RCA.txt')
h_cl_l_load = np.genfromtxt(FOLDER + 'Displacement_CLdx.txt')
h_cl_r_load = np.genfromtxt(FOLDER + 'Displacement_CLsx.txt')
pressure_load = np.genfromtxt(FOLDER + 'Pressure_signal.txt')

#calculate the averages
ca = (lca_load + rca_load) / 2
h_cl = (h_cl_l_load + h_cl_r_load) / 2

#shift the data
shift_idx = np.argmax(ca > 0)
ca = ca[shift_idx:]
h_cl = h_cl[shift_idx:]
pressure_load= pressure_load[shift_idx:len(ca)+shift_idx]

#smooth the data 
ca_smoothed = savgol_filter(ca, 35, 3, axis = 0)
h_cl_smoothed = savgol_filter(h_cl, 105, 3, axis = 0)
pressure_smoothed = savgol_filter(pressure_load, 55, 3, axis = 0)

#calibrate the pressure
pressure_smoothed = pressure_smoothed * 208.74 - 11.82
pressure_load = pressure_load * 208.74 - 11.82

#set up the timesteps
t = np.arange(0, len(ca)/500, 0.002)

step_pressure = np.zeros(len(pressure_smoothed)) + pressure_step
advanced_step_pressure = np.copy(pressure_smoothed)
idx = np.argmax(pressure_smoothed > pressure_step)
advanced_step_pressure[idx:] = pressure_step

#set up the interpolation functions
pres_inter = interpolate.splrep(t, pressure_smoothed)
def pres_smoothed(t):
    return interpolate.splev(t, pres_inter)
pres_inter_load = interpolate.splrep(t, pressure_load)
def pres(t):
    return interpolate.splev(t, pres_inter_load)
pres_step_inter = interpolate.splrep(t, step_pressure)
def pres_step(t):
    return interpolate.splev(t, pres_step_inter)
pres_step_adv_inter = interpolate.splrep(t, advanced_step_pressure)
def pres_step_adv(t):
    return interpolate.splev(t, pres_step_adv_inter)
ca_inter = interpolate.splrep(t, ca_smoothed)
def ca(t):
    return interpolate.splev(t, ca_inter)


#set the inital values
h0 = 0.074 + h_cl_smoothed[0]/1000
velocity = np.gradient(h_cl_smoothed)
dh0 = velocity[0]/(0.002*1000)
X0 = np.array([dh0, h0])

#Define the ODE functions
def ode_normal(X, t):
    U, Y = X
    dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres(t)/rhoL\
                      + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
    dydt = U
    return [dudt, dydt]
def ode_filter(X, t):
    U, Y = X
    dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_smoothed(t)/rhoL\
                      + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
    dydt = U
    return [dudt, dydt]
def ode_step(X, t):
    U, Y = X
    dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_step(t)/rhoL\
                      + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
    dydt = U
    return [dudt, dydt]
def ode_step_adv(X, t):
    U, Y = X
    dudt = (Y)**(-1)*(-g*Y - 12*(l+delta)*mu*U*Y/(rhoL*l*delta**2) + pres_step_adv(t)/rhoL\
                      + 2.0*(l+delta)*sigma*np.cos(ca(t))/(rhoL*l*delta) - U*abs(U))
    dydt = U
    return [dudt, dydt]

#calculate the solutions
solution_normal = odeint(ode_normal, X0, t)
solution_filter = odeint(ode_filter, X0, t)
solution_step = odeint(ode_step, X0, t)
solution_step_adv = odeint(ode_step_adv, X0, t)

#save the calculated data
SAVE = '1500_pa' + os.sep
np.savetxt(SAVE + 'Normal_1500.txt', solution_normal)
np.savetxt(SAVE + 'Filter_1500.txt', solution_filter)
np.savetxt(SAVE + 'Step_1500.txt', solution_step)
np.savetxt(SAVE + 'Step_adv_1500.txt', solution_step_adv)