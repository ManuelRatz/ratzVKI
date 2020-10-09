# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:59:47 2020
@author: koval
@description: All functions for the edge detection.
"""
###############################################################################
# Functions
###############################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os
import cv2
import imageio
from scipy.signal import find_peaks
import Fitting_tool_functions as aFitting


def detect_triggersHS(folder):
    
    def read_lvm(path):
    
    #path_cam = folder + 'test-THS-cam_001.lvm'
        header    = 12  # number of header rows
        value     = 15  # gives the number of the loop
        with open(path) as alldata:
            line = alldata.readlines()[14]
        n_samples = int(line.strip().split('\t')[1])
    
        t = []
        v = []
        for i in range(value):
            with open(path) as alldata:                       #read the data points with the context manager
                lines = alldata.readlines()[header+11+i*(n_samples+11):(header+(i+1)*(n_samples+11))]
            t_temp        = [float(line.strip().split('\t')[0]) for line in lines] 
            v_temp        = [float(line.strip().split('\t')[1]) for line in lines]
            t             = np.append(t,t_temp)
            v             = np.append(v,v_temp)
        
        return t,v
    
    # paths
    path_cam = folder + '/test-HS-cam_001.lvm'
    t_cam,v_cam = read_lvm(path_cam)
    path_pr = folder + '/test-HS-pressure_001.lvm'
    t_pr,v_pr = read_lvm(path_pr)
    path_valve = folder + '/test-HS-valve_001.lvm'
    t_valve,v_valve = read_lvm(path_valve)
    
    
    f_acq = int(len(t_cam)/t_cam[-1])
    idx_start_cam = np.argwhere(v_cam > 3)[0]
    idx_start_valve = np.argwhere(v_valve > 4)[0]
    pressure_signal = v_pr[int(idx_start_valve):]
    frame0 = (t_valve[idx_start_valve]-t_cam[idx_start_cam])*f_acq
    return frame0, pressure_signal


def edge_detection_grad(crop_img,treshold_pos,wall_cut):
    """
    Function to detect horizontal edges
    :crop_img: the cropped image
    :treshold: treshold in order to cut the smaller gradient
    
    """
    grad_img = np.zeros(crop_img.shape)
    y_index  = (np.asarray([])) 
    
    Profiles   = crop_img[:,wall_cut:len(crop_img[1])-wall_cut] # analyse the region where no spots from the wall disturb detection
    Profiles_s = savgol_filter(Profiles, 15, 3, axis =0)        # intensity profile smoothened
    Profiles_d = np.gradient(Profiles_s, axis = 0)              # calculate gradient of smoothend intensity along the vertical direction
    idx_maxima = np.argmax(Profiles_d, axis = 0)                # find positions of all the maxima in the gradient
    #mean_a = np.mean(Profiles_s, axis=0)
    for j in range(Profiles_d.shape[1]):
        # accept only the gradient peaks that are above the threeshold (it avoids false detections lines where the interface is not visible)  
        if Profiles_d[idx_maxima[j],j] > treshold_pos:
            grad_img[idx_maxima[j],j+wall_cut] = 1 # binarisation
            
    y_index, x_index = np.where(grad_img==1)

    # filter out outliers
    y_average  = np.median(y_index)
    for k in range(len(y_index)):
        if np.abs(y_index[k]-y_average)/np.median(y_index)>0.1:
                grad_img[int(y_index[k]),x_index[k]] = 0
    
    return grad_img, y_index, x_index

def fitting_polynom2(grad_img,pix2mm):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    def func(x, a, b, c):
      return a*(x)**2+b*x+c
    
    right_flip_img = np.flipud(grad_img)   # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_y_mm = i_y*pix2mm                    # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm                    # x coordinate of the edge in mm
    img_width_mm = len(grad_img[1])*pix2mm # width of the image in mm
    i_x_shifted = i_x_mm                   # shifting the coordinates
    x = np.linspace(0,img_width_mm,100000)
    popt_cons, _ = curve_fit(func, i_x_shifted, i_y_mm, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
    return func,popt_cons,i_x,i_y,i_x_mm,i_y_mm,x,img_width_mm

def fitting_polynom4(grad_img,pix2mm):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    from scipy.optimize import curve_fit
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = len(grad_img[1])*pix2mm # width of the image in mm
    i_x_shifted = i_x_mm 
    x = np.linspace(0,img_width_mm,100000)
    def func(x, a, b, c, d, e):
      return a*(x)**4+b*x**3+c*x**2+d*x+e
    # fitting
    popt_cons, _ = curve_fit(func, i_x_shifted, i_y_mm, bounds=([-np.inf, -np.inf, -np.inf , -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))

    return func,popt_cons,i_x,i_y,i_x_mm,i_y_mm,x,img_width_mm

def fitting_polynom6(grad_img,pix2mm):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    from scipy.optimize import curve_fit
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = len(grad_img[1])*pix2mm # width of the image in mm
    i_x_shifted = i_x_mm 
    x = np.linspace(0,img_width_mm,100000)
    def func(x, a, b, c, d, e, f, g):
      return a*(x)**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g
    # fitting
    popt_cons, _ = curve_fit(func, i_x_shifted, i_y_mm, bounds=([-np.inf,-np.inf,-np.inf, -np.inf, -np.inf , -np.inf, -np.inf], [np.inf, np.inf,np.inf,np.inf, np.inf, np.inf, np.inf]))

    return func,popt_cons,i_x,i_y,i_x_mm,i_y_mm,x,img_width_mm

def fitting_ellipse(grad_img,pix2mm):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = len(grad_img[1])*pix2mm # width of the image in mm
    i_x_shifted = i_x_mm #shifting the coordinates
    x = np.linspace(0,img_width_mm,100000)
    
    def func(x, xo, a, yo, b):
      return yo-b*(abs(1-((x-xo)/a)**2))**0.5
    # fitting
    BBm   = [(max(x)-min(x))/2*0.98, (max(x)-min(x))/8,min(i_y_mm)*0.1,(max(i_y_mm)-min(i_y_mm))/8]
    BBM   = [(max(x)-min(x))/2*1.02, (max(x)-min(x))*10, max(i_y_mm)*1.5, (max(x)-min(x))*5]
    
    popt_cons, pcov = curve_fit(func, i_x_shifted, i_y_mm, bounds=(BBm,BBM))
    perr = np.sqrt(np.diag(pcov))
    
    if max(perr)>0.15:
        # print(perr,'polinomial interpolation')
        def func(x, *p):
            return np.polyval(p, x)
    
        new_data  = np.stack((np.array(i_x_shifted),np.array(i_y_mm)),axis=-1)
        data_sort = new_data[new_data[:,0].argsort()]
        
        weights  = np.ones(len(data_sort[:,0]))
        popt_cons = np.polyfit(data_sort[:,0], data_sort[:,1],2,w=weights)
        # popt_cons, pcov = curve_fit(func, i_x_shifted, i_y_mm)
    #else: print('ellipse interpolation')
    return func,popt_cons,i_x,i_y,i_x_mm,i_y_mm,x,img_width_mm

def fitting_wall(grad_img,pix2mm):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    from scipy.optimize import curve_fit
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = len(grad_img[1])*pix2mm # width of the image in mm
    x = np.linspace(0,img_width_mm,100000)
    #i_x_smallest = heapq.nsmallest(10, i_x_mm)
    n = 30
    i_x_smallest = np.zeros([n])
    i_y_smallest = np.zeros([n])

    for i in range(n):
        idx = np.argmin(i_x_mm)
        i_x_smallest[i] = np.amin(i_x_mm)
        i_y_smallest[i] = i_y_mm[idx]
        # remove for the next iteration the last smallest value:
        i_x_mm = np.delete(i_x_mm, idx)
    i_x_largest = np.zeros([n])
    i_y_largest = np.zeros([n])

    for i in range(n):
        idx = np.argmax(i_x_mm)
        i_x_largest[i] = np.amax(i_x_mm)
        i_y_largest[i] = i_y_mm[idx]
        # remove for the next iteration the last smallest value:
        i_x_mm = np.delete(i_x_mm, idx)
      

    # def func(x, a, b, c):
    #   return a * np.exp(b * x) + c
    # fitting
    def func_s(x, a, b, c):
      return a*(x)**2+b*x+c
    popt_cons_s, _ = curve_fit(func_s, i_x_smallest, i_y_smallest, maxfev = 10000)
    def func_l(x, a, b, c):
      return a*(x)**2+b*x+c
    popt_cons_l, _ = curve_fit(func_l, i_x_smallest, i_y_smallest, maxfev = 10000)

    return func_s, popt_cons_s ,func_l, popt_cons_l, x

def fitting_advanced(grad_img,pix2mm,l,sigma_f,sigma_y):
    """
    Function to fit a 2nd order polynom for the meniscus with a constrain
    :grad_img: binary image, 1 at the edge of the meniscus
    
    """
    right_flip_img = np.flipud(grad_img) # flipping the image
    i_y, i_x = np.where(right_flip_img==1) # coordinates of the edge
    i_y_mm = i_y*pix2mm # y coordinate of the edge in mm
    i_x_mm = i_x*pix2mm # x coordinate of the edge in mm
    img_width_mm = len(grad_img[1])*pix2mm # width of the image in mm 
   
    X_t=i_x_mm; X_train=np.expand_dims(X_t, axis=1)
    Y_t=i_y_mm; Y_train=np.expand_dims(Y_t, axis=1)
    #y_c=data['y_c'] # This is the clean data.
    X=np.linspace(0, img_width_mm, 1000, endpoint=True)
    X_test=np.expand_dims(X, axis=1)
    # Note that the vectors are augmented to be of sixe n x 1
    
    # This is the Gaussian Fit
    mu_s, cov_s = aFitting.posterior_predictive(X_test, X_train, Y_train,l,sigma_f,sigma_y)
    # This is an example of sampling possibe solutions of the regressions
    #samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    
    # fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
    # plt.scatter(X_train,Y_train,c='white',
    #             marker='o',edgecolor='black',
    #             s=10,label='Data')
    # aFitting.plot_gp(mu_s, cov_s, X_test)
    # plt.xlim([0,5])
    # #plt.ylim([30,40])
    

    return mu_s,i_x,i_y,i_x_mm,i_y_mm,X,img_width_mm

def check_validity6(x,y):
    """
    Function to check if 6th order polynom is good
    
    """
    minima, _ = find_peaks(-1*y)
    maxima, _ = find_peaks(y)
    # plt.figure()
    # plt.plot(minima, y[minima], "x")
    # plt.plot(maxima, y[maxima], "x")
    # plt.plot(y)
    length = len(minima)+ len(maxima)
    
    return length

def vol_average(y,x,img_width_mm):
    """
    Function to calculate average height of the meniscus
    :grad_img: binary image, 1 at the edge of the meniscus
    :x: x coordinate of data for polynomial fitting
    :f: coordinates of the fitted curve
    
    """
    if len(y)==0:
        h_mm = np.nan
    else:
        vol = np.trapz(y, x)  # integrating the fitted line in 2D (?)
        h_mm = vol/(img_width_mm) 
    return h_mm


def contact_angle(y,x,side):
    """
    Function to calculate the contact angle
    :x: x coordinate of data for polynomial fitting
    :f: coordinates of the fitted curve
    :side: 0 for left side, -1 for right side
    :angle: initialisation for array
    
    """
    if side == 0:
        if np.gradient(y,x)[side]< 0:
            grad = abs(np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90-alfa_deg
        else:
            grad = (np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90+alfa_deg
    else:
        if np.gradient(y,x)[side]> 0:
            grad = abs(np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90-alfa_deg
        else:
            grad = (np.gradient(y,x)[side])
            alfa = np.arctan(grad)
            alfa_deg = alfa*180/np.pi
            angle = 90+alfa_deg
    #angle_rad = np.radians(angle)
    return angle


#%%
def animation(GIFNAME,Fol_Out,n_t,n_exp):
    """
    Function to Create a Gif
    :GIFNAME: Name of the Gif
    :Fol_Out: Folder of the images
    :n_t: Number of frames
    
    """
    
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)  
     
    images=[] 
     
    for k in range(n_exp,n_t,1):
    
        MEX= 'Mounting Im '+ str(k)+' of ' + str(n_t)
        print(MEX)
        FIG_NAME=Fol_Out+os.sep+'Step_'+str(k)+'.png'
        images.append(imageio.imread(FIG_NAME))
        # assembly the video
    imageio.mimsave(GIFNAME, images,duration=0.01)
    # import shutil  #  delete a folder and its content
    # shutil.rmtree(Fol_Out)

    MEX='Animation'+GIFNAME+'Ready'
    print(MEX)