import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(y, t):
    dydt = -y + 1.0
    return dydt

t = np.linspace(0, 5)

y0 = 0

y = odeint(model, y0, t)

plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')