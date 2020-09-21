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

FOLDER = 'experimental_data' + os.sep + '1250_pascal'
disp_cl_l = np.genfromtxt(FOLDER + os.sep + 'cl_l_1250.txt')
disp_cl_r = np.genfromtxt(FOLDER + os.sep + 'cl_r_1250.txt')
lca = np.genfromtxt(FOLDER + os.sep + 'lca_1250.txt')
rca = np.genfromtxt(FOLDER + os.sep + 'rca_1250.txt')
pressure = np.genfromtxt(FOLDER + os.sep + 'pressure_1250.txt')
h_avg = np.genfromtxt(FOLDER + os.sep + 'avg_height_1250.txt')

#shift the arrays to have the same starting point not equal to zero
shift = np.argmax(lca > 0)
disp_cl_l = disp_cl_l[shift:]
disp_cl_r = disp_cl_r[shift:]
lca = lca[shift:]
rca = rca[shift:]
h_avg = h_avg[shift:]
pressure = pressure[shift:len(disp_cl_l)+shift]

lca = savgol_filter(lca, 25, 3, axis = 0)
rca = savgol_filter(rca, 25, 3, axis = 0)
disp_cl_r = savgol_filter(disp_cl_r, 25, 3, axis = 0)
disp_cl_l = savgol_filter(disp_cl_l, 25, 3, axis = 0)

t = np.arange(0, len(lca)/500, 0.002)

fig, ax = plt.subplots(figsize = (8,5))
plt.plot(t, ((disp_cl_l+disp_cl_r)/2), label = 'right contact line')
plt.plot(t, h_avg, label = 'average height')
#ax.set_ylim([90,115])
#ax.set_xlim([0, 2.5])
ax.axes.yaxis.set_visible(False)
plt.grid()
plt.plot(t, (lca + rca)/100+0.05, label = 'contact angle')
plt.legend()