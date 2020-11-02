import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

"""
Here different settings for the plots are given.
"""

plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=20)   # fontsize of the figure title
plt.rc('text', usetex=True)      # use latex for the text
plt.rc('font', family='serif')   # serif as text font
plt.rc('axes', grid=True)        # enable the grid
plt.rc('savefig', dpi = 200)     # set the dpi for saving figures

# this is the array of the different pressures that were measured. It is given
# to be able to iterate over all the files in the folders
Pressure = np.array([1000, 1250, 1500])

# set up the for loop
for i in range(0, 3):
    """
    Here the experimental data, calculated acceleration and calculated solutions
    are loaded. The contact line height is averaged and filtered to allow for a 
    better comparison. The data also needs to be shifted to account for the nans
    at the beginning of the files
    """
    # Load the solutions and accelerations
    FOL_IN = '%d_pa' %Pressure[i] + os.sep
    acc_filter = np.genfromtxt(FOL_IN + 'Acc_Filter_%d.txt' %Pressure[i])
    acc_step = np.genfromtxt(FOL_IN +  'Acc_Step_%d.txt' %Pressure[i])
    acc_step_adv = np.genfromtxt(FOL_IN + 'Acc_Step_adv_%d.txt' %Pressure[i])
    sol_filter = np.genfromtxt(FOL_IN + 'Sol_Filter_%d.txt' %Pressure[i])
    sol_step = np.genfromtxt(FOL_IN + 'Sol_Step_%d.txt' %Pressure[i])
    sol_step_adv = np.genfromtxt(FOL_IN + 'Sol_Step_adv_%d.txt' %Pressure[i])
    
    # Load the experimental data
    FOL_IN = '..' + os.sep + 'experimental_data' + os.sep + '%d_pascal' %Pressure[i] + os.sep
    data_cl_l = np.genfromtxt(FOL_IN + 'cl_l_%d.txt' %Pressure[i])
    data_cl_r = np.genfromtxt(FOL_IN + 'cl_r_%d.txt' %Pressure[i])
    data_lca = np.genfromtxt(FOL_IN + 'lca_%d.txt' %Pressure[i])
    data_rca = np.genfromtxt(FOL_IN + 'rca_%d.txt' %Pressure[i])
    avg_height = np.genfromtxt(FOL_IN + 'avg_height_%d.txt' %Pressure[i])
    
    # calculate the average contact line height and contact angle
    ca = data_rca
    
    # shift the data
    shift_idx = np.argmax(ca > 0)
    data_lca = data_lca[shift_idx:]
    data_rca = data_rca[shift_idx:]
    data_cl_r = data_cl_r[shift_idx:]
    ca = ca[shift_idx:]
    avg_height = avg_height[shift_idx:]
    
    # smooth the height and contact angle
    ca = savgol_filter(ca, 25, 3, axis = 0)
    avg_height = savgol_filter(avg_height, 55, 3, axis = 0)
    """
    Here the plots of the data are created in order to compare the different
    results. The pressure signals have already been plotted in the calculation
    of the solutions. The unfiltered pressure is very noisy for 1000 and 1250
    pascal, so it will not be further looked at. For now the plots are as follows:
        - Experimental height vs height predicted from the data
        - All the heights for different pressure inputs and the experimantal height
        - height with contact angle
        - Acceleration for every term with a global scopethis is done for all
          3 pressure signals
        - Acceleration for gravity-pressure and the other terms, this is done 
          for all 3 pressure signals
    """
    # set up the timesteps
    t = np.arange(0, len(ca)/500, 0.002)
    #set up the saving folder for the images
    FOL_OUT = '%d_pa_images' %Pressure[i] + os.sep
    
    # plot the raw contact angles
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, np.cos(data_lca), label = 'cos$(\Theta_L)$')
    plt.plot(t, np.cos(data_rca), label = 'cos$(\Theta_R)$')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('cos($\Theta$)')
    ax.set_xlim([0, 4])
    plt.legend(loc = 'lower right')
    # plt.title('Comparison of the cosine of the left and right contact angle (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'cos_of_ca_comparison_%d.png' %Pressure[i])
    
    # plot the heights for the experimental data and the model
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, sol_filter[:, 1]*1000, label = 'Model Prediction')
    plt.plot(t, avg_height*1000, label = 'Experimental data')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    ax.set_xlim([0, 4])
    ax.set_ylim([avg_height[0]*1000, np.amax(sol_filter[:,1]*1000)+2])
    plt.legend(loc = 'lower right')
    # plt.title('Height comparison for model prediction and experimental data (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Transient_height_comparison_%d.png' %Pressure[i])
    
    # plot the heights for different pressure signals
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, sol_filter[:, 1]*1000, label = 'Filtered Pressure')
    plt.plot(t, sol_step[:, 1]*1000, label = 'Step pressure')
    plt.plot(t, sol_step_adv[:, 1]*1000, label = 'Advanced step pressure')
    plt.plot(t, avg_height*1000, label = 'Experimental data')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$h$[mm]')
    ax.set_xlim([0, 4])
    ax.set_ylim([avg_height[0]*1000, np.amax(sol_step[:,1]*1000)+2])
    plt.legend(loc = 'lower right')
    # plt.title('Height comparison for different pressure signals (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Transient_height_pressure_comparison_%d.png' %Pressure[i])
    plt.close(fig)
    
    # plot the contact angle with the heights
    
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.plot(t, ca*180/np.pi, label = 'Contact angle')
    plt.plot(t, data_cl_r*700, label = 'Contact line height')
    plt.plot(t, avg_height*700, label = 'Average height')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$\Theta$[$^\circ$]')
    #ax.axes.yaxis.set_visible(False)
    ax.set_xlim([0, 4])
    ax.set_ylim([24, 100])
    plt.legend(loc = 'upper right')
    # plt.title('Contact angle compared to average and contact line height (%d Pa)' % Pressure[i])
    plt.savefig(FOL_OUT + 'Contact_angle_comparison_height_%d.png' %Pressure[i])
    
    # plot the accelerations for a global scope (filter)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(t, acc_filter[:, 0], label = 'Gravity')
    plt.plot(t, acc_filter[:, 1], label = 'Viscosity')
    plt.plot(t, acc_filter[:, 2], label = 'Pressure')
    plt.plot(t, acc_filter[:, 3], label = 'Surface Tension')
    plt.plot(t, acc_filter[:, 4], label = '$u^2$')
    plt.plot(t, acc_filter[:, 5], label = 'Total')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    ax.set_xlim([0, 4])
    ax.set_ylim([-17, 11])
    plt.legend(ncol = 3, loc = 'lower center')
    # plt.title('Accelerations for the filtered pressure (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Acceleration_filtered_pressure_%d.png' % Pressure[i])
    plt.close(fig)
    
    # plot of the zoomed acceleration (filter)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(t, acc_filter[:, 0] + acc_filter[:, 2], label = 'Gravity + pressure')
    plt.plot(t, acc_filter[:, 1], label = 'Viscosity')
    plt.plot(t, acc_filter[:, 3], label = 'Surface Tension')
    plt.plot(t, acc_filter[:, 4], label = '$u^2$')
    plt.plot(t, acc_filter[:, 5], label = 'Total')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    ax.set_xlim([0, 4])
    ax.set_ylim([-1.5, 1])
    plt.legend(loc = 'lower right', ncol = 2)
    # plt.title('Zoomed in accelerations for filtered pressure (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Zoomed_accelerations_filtered_pressure_%d.png' %Pressure[i])
    plt.close(fig)
    
    # plot the accelerations for a global scope (step)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(t, acc_step[:, 0], label = 'Gravity')
    plt.plot(t, acc_step[:, 1], label = 'Viscosity')
    plt.plot(t, acc_step[:, 2], label = 'Pressure')
    plt.plot(t, acc_step[:, 3], label = 'Surface Tension')
    plt.plot(t, acc_step[:, 4], label = '$u^2$')
    plt.plot(t, acc_step[:, 5], label = 'Total')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    ax.set_xlim([0, 4])
    ax.set_ylim([-17, 11])
    plt.legend(ncol = 3, loc = 'lower center')
    # plt.title('Accelerations for the step pressure (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Acceleration_step_pressure_%d.png' % Pressure[i])
    plt.close(fig)
    
    # plot of the zoomed acceleration (step)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(t, acc_step[:, 0] + acc_step[:, 2], label = 'Gravity + pressure')
    plt.plot(t, acc_step[:, 1], label = 'Viscosity')
    plt.plot(t, acc_step[:, 3], label = 'Surface Tension')
    plt.plot(t, acc_step[:, 4], label = '$u^2$')
    plt.plot(t, acc_step[:, 5], label = 'Total')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    ax.set_xlim([0, 4])
    ax.set_ylim([-1.5, 1])
    plt.legend(loc = 'lower right', ncol = 2)
    # plt.title('Zoomed in accelerations for step pressure (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Zoomed_accelerations_step_pressure_%d.png' %Pressure[i])
    plt.close(fig)
    
    # plot the accelerations for a global scope (step_adv)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(t, acc_step_adv[:, 0], label = 'Gravity')
    plt.plot(t, acc_step_adv[:, 1], label = 'Viscosity')
    plt.plot(t, acc_step_adv[:, 2], label = 'Pressure')
    plt.plot(t, acc_step_adv[:, 3], label = 'Surface Tension')
    plt.plot(t, acc_step_adv[:, 4], label = '$u^2$')
    plt.plot(t, acc_step_adv[:, 5], label = 'Total')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    ax.set_xlim([0, 4])
    ax.set_ylim([-17, 11])
    plt.legend(ncol = 3, loc = 'lower center')
    # plt.title('Accelerations for the advanced step pressure (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Acceleration_step_adv_pressure_%d.png' % Pressure[i])
    plt.close(fig)
    
    # plot of the zoomed acceleration (step_adv)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(t, acc_step_adv[:, 0] + acc_step_adv[:, 2], label = 'Gravity + pressure')
    plt.plot(t, acc_step_adv[:, 1], label = 'Viscosity')
    plt.plot(t, acc_step_adv[:, 3], label = 'Surface Tension')
    plt.plot(t, acc_step_adv[:, 4], label = '$u^2$')
    plt.plot(t, acc_step_adv[:, 5], label = 'Total')
    ax.set_xlabel('$t$[s]')
    ax.set_ylabel('$a$[m/s$^2$]')
    ax.set_xlim([0, 4])
    ax.set_ylim([-1.5, 1])
    plt.legend(loc = 'lower right', ncol = 2)
    # plt.title('Zoomed in accelerations for advanced step pressure (%d Pa)' %Pressure[i])
    plt.savefig(FOL_OUT + 'Zoomed_accelerations_step_adv_pressure_%d.png' %Pressure[i])
    plt.close(fig)
    
    """
    These are experimental plots to compare the velocity of the contact line to the contact angle
    """
    """
    #calculate the velocities
    deriv_cl = np.gradient(h_cl) / 0.002
    deriv_avg = np.gradient(avg_height) / 0.002
    
    #smooth the gradients
    deriv_cl_smoothed = savgol_filter(deriv_cl, 125, 2, axis = 0)
    deriv_avg_smoothed = savgol_filter(deriv_avg, 135, 2, axis = 0)
    
    #create the plot
    fig, ax = plt.subplots(figsize = (8, 5))
    plt.scatter(deriv_cl_smoothed, ca*180/np.pi)
    plt.legend()
    """
    
    
    
    
    
    
    
    
    
    
    
    
    