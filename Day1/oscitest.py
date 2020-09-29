import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


## Hello Manuel, this is Miguel. Put more comments plz

def osci(y, t):
    ye = y[0]
    ze = y[1]
    yn = y[2]
    zn = y[3]
    
    dyedt = ze
    dzedt = -np.sin(ye)
    
    dyndt = zn
    dzndt = -yn
    
    ret = [dyedt, dzedt, dyndt, dzndt]
    return ret

y0 = [np.pi/8, 0, np.pi/8, 0]

t = np.linspace(0, 1, 150)

y = odeint(osci, y0, t)

plt.plot(t, y[:, 0])
plt.plot(t, y[:, 2])
plt.xlabel('time')
plt.ylabel('y(t)')