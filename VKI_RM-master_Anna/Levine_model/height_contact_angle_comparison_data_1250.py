import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

#=============================================================================
# Load the Data files and process so they have the same length and correct scaling
#=============================================================================

FOLDER = 'experimental_data' + os.sep + '1500_pascal'
disp_cl_l = np.genfromtxt(FOLDER + os.sep + 'Displacement_CLsx.txt')
disp_cl_r = np.genfromtxt(FOLDER + os.sep + 'Displacement_CLdx.txt')
lca = np.genfromtxt(FOLDER + os.sep + 'LCA.txt')
rca = np.genfromtxt(FOLDER + os.sep + 'RCA.txt')
pressure = np.genfromtxt(FOLDER + os.sep + 'Pressure_signal.txt')
h_avg = np.genfromtxt(FOLDER + os.sep + 'Displacement.txt')

#shift the arrays to have the same starting point not equal to zero
shift = np.argmax(lca > 0)
disp_cl_l = disp_cl_l[shift:]
disp_cl_r = disp_cl_r[shift:]
lca = lca[shift:]
rca = rca[shift:]
h_avg = h_avg[shift:]
pressure = pressure[shift:len(disp_cl_l)+shift]

lca = savgol_filter(lca, 15, 3, axis = 0)
rca = savgol_filter(rca, 15, 3, axis = 0)
disp_cl_r = savgol_filter(disp_cl_r, 15, 3, axis = 0)
disp_cl_l = savgol_filter(disp_cl_l, 15, 3, axis = 0)

t = np.arange(0, len(lca)/500, 0.002)

fig, ax = plt.subplots(figsize = (8,5))
plt.plot(t, ((disp_cl_l+disp_cl_r)/2)+74, label = 'right contact line')
plt.plot(t, h_avg+74, label = 'average height')
#ax.set_ylim([90,115])
#ax.set_xlim([0, 2.5])
ax.axes.yaxis.set_visible(False)
plt.grid()
plt.plot(t, (lca + rca)*20/2+70, label = 'contact angle')
plt.legend()