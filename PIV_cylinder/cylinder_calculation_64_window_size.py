import numpy as np
import matplotlib.pyplot as plt

#define the radius of the cylinder
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

#filter out the cylinder
inv = rho < radius
rho[inv] = np.nan

#calculate the velocities in polar coordinates
ang_vel = angular_velocity(rho, angle)
rad_vel = radial_velocity(rho, angle)

#calculate the velocities in cartesian coordinates
v_x = np.cos(angle) * radial_velocity(rho, angle) + np.sin(angle) * angular_velocity(rho, angle)
v_y = np.sin(angle) * radial_velocity(rho, angle) - np.cos(angle) * angular_velocity(rho, angle)
mag = np.sqrt(v_x**2+v_y**2)


#plot the velocity field
fig, ax = plt.subplots(figsize=(8, 5))
plt.contourf((cartesian_grid[0,:,:]+500), (cartesian_grid[1,:,:]+500), mag)
plt.quiver((cartesian_grid[0,:,:]+500), (cartesian_grid[1,:,:]+500), v_x, v_y, scale = 15)
plt.clim(0, 10)
plt.colorbar()
plt.title('Theoretical velocity field')
ax.set_aspect('equal')  # Set equal aspect ratio
ax.set_xlabel('$x[mm]$', fontsize=18)
ax.set_ylabel('$y[mm]$', fontsize=18)

plt.savefig('Theoretical_Velocity_Field_winsize_64.png', dpi = 100)
plt.show()
plt.close('all')

######################################################################
##extract the velocity profile shortly befor the cylinder#############
######################################################################
"""
x_vel_plot = v_x[:, 10] #x velocity 116 mm before the cylinder
x_val = np.arange(-484, 492, 16) #corresponding y values for plotting
x_vel_plot = x_vel_plot / np.max(x_vel_plot)

plt.plot(x_val, x_vel_plot,  'ko')
"""