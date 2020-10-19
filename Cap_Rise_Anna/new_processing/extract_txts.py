"""
@author: ratz
@description: extract the 6 signals out of the .txt file that was created in the
post processing. This is done for all 3 pressure signals

"""

import numpy as np  # for operations with arrays
import os           # for creating data paths

def readResultsFile(path):
    # function from Anna that extracts the columns of the txt file and saves them in 1d arrays
    with open(path) as alldata: #read the data points with the context manager
        lines = alldata.readlines()[21:]
        t_a = np.array([float(line.strip().split()[0]) for line in lines])
        h = np.array([float(line.strip().split()[1]) for line in lines])
        h_l = np.array([float(line.strip().split()[2]) for line in lines])
        h_r = np.array([float(line.strip().split()[3]) for line in lines])
        lca = np.array([float(line.strip().split()[4]) for line in lines])
        rca = np.array([float(line.strip().split()[5]) for line in lines])
        p = np.array([float(line.strip().split()[6]) for line in lines])
    return t_a, h, h_l, h_r,lca,rca, p 

# set up the pressure array for naming purposes
pressure_array = np.array([1000, 1250, 1500])
for i in range(2, len(pressure_array)):
    
    # create the input path
    # path = 'experimental_data' + os.sep
    time, height, cl_l, cl_r, lca_deg, rca_deg, pres = readResultsFile('test.txt')
    # Shift the data to account for the NaNs at the beginning
    idx = np.argmax(cl_l>0)
    height = height[idx:]
    cl_l = cl_l[idx:]
    cl_r = cl_r[idx:]
    lca_deg = lca_deg[idx:]
    rca_deg = rca_deg[idx:]
    pres = pres[idx:]
    
    # create the output path
    path = 'data_1500_pa' + os.sep 
    # Save the data
    np.savetxt(path + 'cl_l_%d.txt' %pressure_array[i], cl_l)
    np.savetxt(path + 'cl_r_%d.txt' %pressure_array[i], cl_r)
    lca_rad = lca_deg * np.pi/180
    rca_rad = rca_deg * np.pi/180
    np.savetxt(path + 'lca_%d.txt' %pressure_array[i], lca_rad)
    np.savetxt(path + 'rca_%d.txt' %pressure_array[i], rca_rad)
    np.savetxt(path + 'pressure_%d.txt' %pressure_array[i], pres)
    np.savetxt(path + 'avg_height_%d.txt' %pressure_array[i], height)