import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy.signal import savgol_filter
pressure_step = 662.6799

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

#=============================================================================
# Load the Data files and process so they have the same length and correct scaling
#=============================================================================

FOLDER = '..' + os.sep + 'experimental_data' + os.sep + '1000_pascal'
disp_cl_l = np.genfromtxt(FOLDER + os.sep + 'cl_l_1000.txt')
disp_cl_r = np.genfromtxt(FOLDER + os.sep + 'cl_r_1000.txt')
lca = np.genfromtxt(FOLDER + os.sep + 'lca_1000.txt')
rca = np.genfromtxt(FOLDER + os.sep + 'rca_1000.txt')
pressure = np.genfromtxt(FOLDER + os.sep + 'pressure_1000.txt')

#shift the arrays to have the same starting point not equal to zero
shift = np.argmax(lca > 0)
disp_cl_l = disp_cl_l[shift:]
disp_cl_r = disp_cl_r[shift:]
lca = lca[shift:]
rca = rca[shift:]
pressure = pressure[shift:len(disp_cl_l)+shift]

#offset = 0 * np.pi / 180
#smooth all the data
pressure_filter = savgol_filter(pressure, 55, 2, axis =0)
lca_filter = savgol_filter(lca, 55, 3, axis = 0)
rca_filter = savgol_filter(rca, 55, 3, axis = 0)
disp_cl_r_filter = savgol_filter(disp_cl_r, 105, 3, axis = 0)
disp_cl_l_filter = savgol_filter(disp_cl_l, 105, 3, axis = 0)
#calculate the mean contact angle and height
ca_filter = (lca_filter + rca_filter) / 2
disp_cl_filter = (disp_cl_r_filter + disp_cl_l_filter) / 2

#filter the pressure
pressure_filtered = savgol_filter(pressure, 55, 2, axis = 0)

#calculate the step response
step_pressure = np.zeros(len(pressure)) + pressure_step
advanced_step_pressure = np.copy(pressure_filtered)
idx = np.argmax(pressure_filtered > pressure_step)
advanced_step_pressure[idx:] = pressure_step

#set up the timesteps
t = np.arange(0, len(lca)/500, 0.002)

#interpolate the pressure signals
inter_pressure = interpolate.splrep(t, pressure)
inter_pressure_filtered = interpolate.splrep(t, pressure_filtered)
inter_pressure_step = interpolate.splrep(t, step_pressure)
inter_pressure_adv_step = interpolate.splrep(t, advanced_step_pressure)
#interpolate the contact angle
interpolated_ca = interpolate.splrep(t, ca_filter)


#define the pressure functions and contact angle functions
def pres(t):
    p_interpolated = interpolate.splev(t, inter_pressure)
    return p_interpolated

def pres_filter(t):
    p_interpolated = interpolate.splev(t, inter_pressure_filtered)
    return p_interpolated
    
def pres_step(t):
    p_interpolated = interpolate.splev(t, inter_pressure_step)
    return p_interpolated

def pres_step_adv(t):
    p_interpolated = interpolate.splev(t, inter_pressure_adv_step)
    return p_interpolated

def ca_func(t):
    ca_interpolated = interpolate.splev(t, interpolated_ca)
    return ca_interpolated

#compare the results for the different pressures
fig, ax = plt.subplots(figsize = (8,5))
plt.plot(t, pres(t), label = 'Unfiltered')
plt.plot(t, pres_filter(t), label = 'Filtered')
plt.plot(t, pres_step(t), label = 'Step')
plt.plot(t, pres_step_adv(t), label = 'Advanced step response')
plt.title('Comparison of different pressure signals', fontsize = 20)
plt.legend()


g      = 9.81;         # Gravity (m/s2)

# Water
rhoL   = 1000;         #Fluid Density (kg/m3)
mu     = 8.90*10**(-4); # Fuid dynamic viscosity (Pa*s)
sigma  = 0.07197;      # surface tension (N/m)

R      = 0.0025; # Tube radius/side (m)
n      = 100      # trasversal width of the channel in terms of n*D
C1     = 2*(n/(n+1))  # corrective factor for parallel plates
# =============================================================================
## Solving
# =============================================================================      
# u is the velocity of the rise, 
# p is the pressure in the tank
# y is the rise

h0 = disp_cl_filter[0] #the offset is already included in the data
dh0 = (disp_cl_filter[1] - disp_cl_filter[0])/(0.002)


Xzero = np.array([dh0, h0])
def deriv(X,t):
    U, Y  = X 
    dudt = (Y+37./36*R)**(-1)*(-(8*mu/(rhoL*((R*C1)**2)))*U*(Y+0.25*R) - g*Y + (pres(t))/rhoL +(2*sigma/(rhoL*R*C1/2))\
                               *np.cos(ca_func(t))-7./6*(U)*abs(U))
    dydt = U
    return [dudt, dydt]
solvS = integrate.odeint(deriv, Xzero, t)

#calculate the different acceleration terms
pressure_term = (pressure) / rhoL * (solvS[:,1] + 37. * R / 36)**(-1)
surface_tension_term = 2 * sigma * np.cos(ca_filter) / (rhoL * R * C1/2) * (solvS[:,1] + 37. * R / 36)**(-1)
gravity_term = -g * solvS[:,1] * (solvS[:,1] + 37. * R / 36)**(-1)
visc_term = -8*mu*solvS[:,1]*solvS[:,0] / (rhoL * R**2 * C1**2) * (solvS[:,1] + 37. * R / 36)**(-1)
corr_term_1 = -2*mu*R*solvS[:,0] / (rhoL * R**2 * C1**2) * (solvS[:,1] + 37. * R / 36)**(-1)
corr_term_2 = -7. * solvS[:,0]*abs(solvS[:,0])/6 * (solvS[:,1] + 37. * R / 36)**(-1)
total = pressure_term + surface_tension_term + gravity_term + visc_term + corr_term_1 + corr_term_2

#plots of the acceleration terms
fig, ax = plt.subplots(figsize=(8,5))
ax.set_ylim([-1, 1])
plt.plot(t, surface_tension_term, label = 'Surface tension')
plt.plot(t, visc_term, label = 'Viscous term')
plt.plot(t, (pressure_term+gravity_term), label = 'Grav+Pres')
plt.plot(t, corr_term_1, label = 'Correction term, $\mu$')
plt.plot(t, corr_term_2, label = 'Correction term, $U^2$')
plt.plot(t, total, label = 'Total')
plt.grid()
plt.legend()

#plots of the acceleration terms
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(t, surface_tension_term, label = 'Surface tension')
plt.plot(t, pressure_term, label = 'Pressure term')
plt.plot(t, gravity_term, label = 'Gravity term')
plt.plot(t, visc_term, label = 'Viscous term')
plt.plot(t, total, label = 'Total acceleration')
plt.plot(t, corr_term_1, label = 'Correction term, $\mu$')
plt.plot(t, corr_term_2, label = 'Correction term, $U^2$')
plt.grid()
plt.legend()

#plot the heights
fig, ax = plt.subplots(figsize=(8,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
ax.set_xlabel('$t[s]$', fontsize=20)
ax.set_ylabel('$h[mm]$', fontsize=20)
plt.plot(t, solvS[:, 1]*1000, label = 'Model prediction')
plt.plot(t, (disp_cl_l_filter)*1000, label='Experimental Data')
plt.legend(fontsize = 20)
plt.grid()
plt.title('Height comparison over time', fontsize = 20)
plt.savefig('Comparison of the different transient heights')























