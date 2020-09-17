import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def mtest(y, t):
    if (t<10):
        u = 0
    else:
        u = 2
    dydt = (-y + u)/5.0
    return dydt
    
y0 = 1

t = np.linspace(0, 40)

y = odeint(mtest, y0, t)

plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')