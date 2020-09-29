import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

n=200 # amont of samples
t = np.arange(0, n/50, 1/50) # set up the timesteps
data = np.cos(t*np.pi) # create data
data = data + np.random.random(n) * 0.2 # make data noisy

"""
This is testing done on the different settings of firwin.
We are looking at an artifical signal, a cos(t) that has noise.
We are taking 200 samples in total over 4 seconds, meaning our sampling rate
fs = 50. The frequency of our oscillation  is 2 seconds, meaning 0.5 Hz which
is supposed to be the cutoff frequency
"""

# make a lowpass filter
fil = scipy.signal.firwin(20, cutoff=2, window='hanning', fs = 50)

# filter the data with filtfilt
data_filt = scipy.signal.filtfilt(b = fil, a = [1], x = data)

# create a plot to compare the filtered to the unfiltered data
fig, ax = plt.subplots(figsize = (8,5))
plt.plot(t, data, label = 'Unfiltered', c = 'b')
plt.plot(t, data_filt, label = 'filtered', c = 'c')
ax.set_xlabel('t[s]')
ax.set_ylabel('cos(t)')
ax.set_xlim([0, 4])
plt.legend()