import numpy as np
import os
import matplotlib.pyplot as plt
#plot the pressure signals against each other
"""
fig, ax = plt.subplots(figsize = (8, 5))
plt.plot(t, pres(t), label = 'Unfiltered')
plt.plot(t, pres_smoothed(t), label = 'Filtered')
plt.plot(t, pres_step(t), label = 'Step')
plt.plot(t, pres_step_adv(t), label = 'Advanced step')
ax.set_xlabel('$t$[s]', fontsize=20)
ax.set_ylabel('$p$[Pa]', fontsize=20)
ax.set_xlim([0, 4])
ax.set_ylim([900, 1010])
plt.legend(fontsize = 15, loc = 'lower right')
plt.title('Comparison of different pressure signals', fontsize = 20)
plt.grid()
plt.savefig(FOLDER_OUT + 'Pressure_signals_1500.png', dpi = 100)
"""

#plot the heights against each other
"""
fig, ax = plt.subplots(figsize = (8, 5))O
plt.plot(t, solution_normal[:, 1]*1000, label = 'Filtered/unfiltered model')
plt.plot(t, solution_step[:, 1]*1000, label = 'Step response')
plt.plot(t, solution_step_adv[:, 1]*1000, label = 'Advanced step response')
plt.plot(t, h_cl_smoothed+74, label = 'Experimental data')
ax.set_xlabel('$t$[s]', fontsize=20)
ax.set_ylabel('$h$[mm]', fontsize=20)
ax.set_xlim([0, 4])
ax.set_ylim([77, 135])
plt.legend(fontsize = 15, loc = 'lower right')
plt.title('Height comparison for different pressure inputs', fontsize = 20)
plt.grid()
plt.savefig(FOLDER_OUT + 'Transient_height_1500.png', dpi = 100)
"""


"""
In the Latex document make a not that the filtered and unfiltered version of
the pressure are the same because odeint has a filter built in. To save time
and space the unfiltered version will not be considered from now on
"""

"""
Here the accelerations are calculated for each of the 3 configurations
For ease of access and comparison they are stored in a matrix, with each column
representing one of the acceleration terms. The order is as follows:
    0 - gravity             (this will for the current model always be -g)
    1 - viscous term        (this is still not 100% safe, see the meeting with domenico tomorrow, delete after 23.09.)
    2 - pressure term
    3 - contact angle/surface tension term
    4 - velocity squared term
    5 - total acceleration
"""

#calculate the acceleration terms
pressure_names = ['filter', 'step', 'step_adv']
#initialize the acceleration matrices
acc_filter = acc_step = acc_step_adv = np.zeros((solution_normal.shape[0], 6))
for i in range(0, len(pressure_names)):
    #assign the names for the data matrix and solution
    array_name = 'acc_'+pressure_names[i]
    solution_name = 'solution_' + pressure_names[i]
    U = solution_name[:, 0]
    Y = solution_name[:, 1]
    #calculate all the accelerations
    # array_name[:, 0] = -g
    # array_name[:, 1] = -12 * (l + delta)*mu*
    # array_name[:, 2] = 
    # array_name[:, 3] = 
    # array_name[:, 4] = 
    # array_name[:, 5] = 