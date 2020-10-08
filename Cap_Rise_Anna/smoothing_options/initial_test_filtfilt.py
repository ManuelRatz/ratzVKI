"""
@author: ratz
@description: initial testing done with firwin and filtfilt for a test signal
"""

import numpy as np               # for numerical calculations with arrays
import scipy.signal as sci       # for the frequency based filtering
import matplotlib.pyplot as plt  # for plotting things

n=200 # amont of samples
fs = 50 # sampling frequency
t = np.arange(0, n/fs, 1/fs) # set up the timesteps
data = np.cos(t*np.pi) # create data
data = data + np.random.random(n) * 0.2 # make data noisy

# define the observation windows in the time domain
fil = sci.firwin(20, 100/fs*2 , window='hamming', fs = 50)

# filter the data with filtfilt
data_filt = sci.filtfilt(b = fil, a = [1], x = data)

# create a plot to compare the filtered to the unfiltered data
fig, ax = plt.subplots(figsize = (8,5))
plt.plot(t, data, label = 'Unfiltered', c = 'b')
plt.plot(t, data_filt, label = 'filtered', c = 'c')
ax.set_xlabel('t[s]')
ax.set_ylabel('cos(t)')
ax.set_xlim([0, 1])
plt.legend()