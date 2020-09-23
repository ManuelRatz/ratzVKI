import numpy as np
import matplotlib.pyplot as plt
import os

def readResultsFile(path):
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

path = 'experimental_data' + os.sep

time, height, cl_l, cl_r, lca_deg, rca_deg, pres = readResultsFile(path + '2020-08-25-1000Pa_A.txt')

np.savetxt(path + os.sep + '1000_pascal' + os.sep + 'cl_l_1000.txt', cl_l)
np.savetxt(path + os.sep + '1000_pascal' + os.sep + 'cl_r_1000.txt', cl_r)
lca_rad = lca_deg * np.pi/180
rca_rad = rca_deg * np.pi/180
np.savetxt(path + os.sep + '1000_pascal' + os.sep + 'lca_1000.txt', lca_rad)
np.savetxt(path + os.sep + '1000_pascal' + os.sep + 'rca_1000.txt', rca_rad)
np.savetxt(path + os.sep + '1000_pascal' + os.sep + 'pressure_1000.txt', pres)
np.savetxt(path + os.sep + '1000_pascal' + os.sep + 'avg_height_1000.txt', height)

path = 'experimental_data' + os.sep

time, height, cl_l, cl_r, lca_deg, rca_deg, pres = readResultsFile(path + '2020-08-25-1250Pa_A.txt')

np.savetxt(path + os.sep + '1250_pascal' + os.sep + 'cl_l_1250.txt', cl_l)
np.savetxt(path + os.sep + '1250_pascal' + os.sep + 'cl_r_1250.txt', cl_r)
lca_rad = lca_deg * np.pi/180
rca_rad = rca_deg * np.pi/180
np.savetxt(path + os.sep + '1250_pascal' + os.sep + 'lca_1250.txt', lca_rad)
np.savetxt(path + os.sep + '1250_pascal' + os.sep + 'rca_1250.txt', rca_rad)
np.savetxt(path + os.sep + '1250_pascal' + os.sep + 'pressure_1250.txt', pres)
np.savetxt(path + os.sep + '1250_pascal' + os.sep + 'avg_height_1250.txt', height)