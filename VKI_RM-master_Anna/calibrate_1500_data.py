import numpy as np
import os
import matplotlib.pyplot as plt

FOLDER = 'experimental_data' + os.sep + '1500_pascal_backup' + os.sep
h_cl_l_load = np.genfromtxt(FOLDER + 'Displacement_CLdx.txt')
h_cl_r_load = np.genfromtxt(FOLDER + 'Displacement_CLsx.txt')
pressure_load = np.genfromtxt(FOLDER + 'Pressure_signal.txt')
disp_load = np.genfromtxt(FOLDER + 'Displacement.txt')
lca_load = np.genfromtxt(FOLDER + 'LCA.txt')
rca_load = np.genfromtxt(FOLDER + 'RCA.txt')

#shift the data
idx = np.argmax(h_cl_l_load > 0)
disp_load = disp_load[idx:]
h_cl_l_load = h_cl_l_load[idx:]
h_cl_r_load = h_cl_r_load[idx:]
lca_load = lca_load[idx:]
rca_load = rca_load[idx:]
pressure_load = pressure_load[idx:]

#convert to mm
disp_load = 0.074 + disp_load / 1000
h_cl_l_load = 0.074 + h_cl_l_load / 1000
h_cl_r_load = 0.074 + h_cl_r_load / 1000

#calibrate the pressure
pressure_load = pressure_load * 208.74 - 11.82

FOL_OUT = 'experimental_data' + os.sep + '1500_pascal' + os.sep
np.savetxt(FOL_OUT + 'cl_l_1500.txt', h_cl_l_load)
np.savetxt(FOL_OUT + 'cl_r_1500.txt', h_cl_r_load)
np.savetxt(FOL_OUT + 'lca_1500.txt', lca_load)
np.savetxt(FOL_OUT + 'rca_1500.txt', rca_load)
np.savetxt(FOL_OUT + 'pressure_1500.txt', pressure_load)
np.savetxt(FOL_OUT + 'avg_height_1500.txt', disp_load)