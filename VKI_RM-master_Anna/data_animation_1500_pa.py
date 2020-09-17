import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate

FOLDER = 'experimental_data'
#Read all the txts into variables
disp = np.genfromtxt(FOLDER + os.sep + 'Displacement.txt')
disp_cl_l = np.genfromtxt(FOLDER + os.sep + 'Displacement_CLsx.txt')
disp_cl_r = np.genfromtxt(FOLDER + os.sep + 'Displacement_CLdx.txt')
lca = np.genfromtxt(FOLDER + os.sep + 'LCA.txt')
rca = np.genfromtxt(FOLDER + os.sep + 'RCA.txt')
pressure = np.genfromtxt(FOLDER + os.sep + 'Pressure_signal.txt')

#shift the arrays to have the same starting point not equal to zero
shift = np.argmax(disp > 0)
disp = disp[shift:]
disp_cl_l = disp_cl_l[shift:]
disp_cl_r = disp_cl_r[shift:]
lca = lca[shift:]
rca = rca[shift:len(disp)+shift]
pressure = pressure[shift:len(disp)+shift]

#Set up the timesteps
t = np.arange(0, len(disp)/500, 0.002)
#Set up the finer grid to interpolate
refinement_factor = 5
t_fine = np.arange(0, (len(disp)/500)-(refinement_factor-1)*0.002/refinement_factor, 0.002/refinement_factor)

#convert pressure signal to actual pressure
pres_real = 100000 + pressure * 208.74 - 11.82


filterparameter =55
smoothingpolynome = 3
#smooth all the data
pres_fil_real = savgol_filter(pres_real, 25, 3, axis =0)
lca_filter = savgol_filter(lca, filterparameter, smoothingpolynome, axis = 0)
rca_filter = savgol_filter(rca, filterparameter, smoothingpolynome, axis = 0)
disp_cl_r = savgol_filter(disp_cl_r, filterparameter, 3, axis = 0)
disp_cl_l = savgol_filter(disp_cl_l, filterparameter, 3, axis = 0)


#interpolate the pressure
f_pres = interpolate.interp1d(t, pres_fil_real)
pres_fine = f_pres(t_fine)
#interpolate the contact angle
f_lca = interpolate.interp1d(t, lca_filter)
lca_fine = f_lca(t_fine)
f_rca = interpolate.interp1d(t, rca_filter)
rca_fine = f_rca(t_fine)

plt.plot(t, pres_real)
"""
end = 10
start = 0
disp_cl = (disp_cl_l + disp_cl_r) / 2
fig, ax = plt.subplots(figsize = (8, 5))
init_vel_hand_1 = (disp_cl[1]-disp_cl[0])/2
init_vel_hand_2 = (disp_cl[2]-disp_cl[1])/2
plt.scatter(t[start:end], disp_cl[start:end])
f_pres = interpolate.splrep(t, disp_cl)
disp_of_time = interpolate.splev(t, f_pres)
plt.plot(t, disp_of_time)

#Plot the pressure 
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Pressure over time')
#ax.set_xlim([0.7, 0.71])
#ax.set_ylim(971.5, 972.5)
ax.set_xlabel('$time[s]$')
ax.set_ylabel('$p[pa]$')
plt.grid()
#plt.plot(t, pressure[:len(disp)])
plt.plot(t, pres_fil_real[:len(disp)])
plt.scatter(t_fine, pres_fine)
plt.savefig('Pressure_over_time.png', dpi = 100)
plt.show()
plt.close('all')
"""
"""
#plot the left contact angle
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Contact angle over time')
#ax.set_xlim([0, 0.1])
#ax.set_ylim(1.8, 2)
ax.set_xlabel('$time[s]$')
ax.set_ylabel('$\Theta[radians]$')
plt.grid()
#plt.plot(t, lca)
#plt.scatter(t, np.cos(lca_filter))
plt.plot(t_fine, lca_fine*180/np.pi)
plt.savefig('Contact_angle_over_time.png', dpi = 100)
plt.show()
plt.close('all')
"""
"""
#Plot the left contact angle
fig, ax = plt.subplots(figsize = (8, 5))
plt.title('Average height over time')
ax.set_xlim([0,3])
ax.set_ylim(0,2)
plt.grid()
plt.plot(t, lca_smooth[:1500])
plt.savefig('Left_CA.png', dpi = 100)
plt.show()
plt.close('all')
"""