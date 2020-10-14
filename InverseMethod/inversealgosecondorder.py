import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#Load Data
data = np.loadtxt('exportdatasecord.csv', delimiter=',')
j = 0
temp_arr = []

#Calculate Mean Square Root
def meanroot(a):
    y = odeint(ode, y0, t, args=(a,))
    tempsum = np.sum(y-data[0])**2
    return (np.sqrt(tempsum))
#Define ODE
def ode(ye, t, a):
    y = ye[0]
    z = ye[1]
    dydt = z
    dzdt = - z / a[0] - a[1] * y
    return[dydt, dzdt]

#Set initial values
y0 = np.array([1, 0])

#Initial Guess
a0 = np.array([6, 2])
ae = np.array([4, 1])
#create t points
t = np.linspace(0, 20, len(data))

#while (tempsum > 0.01):
solution = minimize(meanroot, a0)

a_sol = solution.x
y_sol = odeint(ode, y0, t, args=(a_sol,))

plt.plot(t, data[:, 1])
# plt.plot(t, y_sol[:,0])
"""
GIFNAME = 'animation.gif'
images = []
for k in range(1, j, 1):
    FIG_NAME = 'graph_images' + os.sep + 'Im%03d' % (k + 1) + '.png'
    images.append(imageio.imread(FIG_NAME))
for k in range(1, 10, 1):
    FIG_NAME = 'graph_images' + os.sep + 'Im%03d' % (j-2) + '.png'
    images.append(imageio.imread(FIG_NAME))


# Now we can assembly the video and clean the folder of png's (optional)
imageio.mimsave(GIFNAME, images, duration=0.25)

arr = np.array(temp_arr)
iterations = np.zeros(51)
for i in range(0, 17):
    iterations[3*i] = i+1
    iterations[3*i+1] = i+1
    iterations[3*i+2] = i+1
fig, ax = plt.subplots(figsize=(8, 5))

plt.plot(iterations, arr, label = 'Cost Function')

plt.rc('text', usetex=True)  # This is Miguel's customization
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_ylim([1e-3, 1e23])
ax.set_xlim([1, 17])
ax.set_yscale('log')
ax.set_xticks(np.arange(1, 17.1, 1))

ax.set_xlabel('Iteration', fontsize=18)
ax.set_ylabel('Cost Function', fontsize=18)
plt.title('Cost Function over the iterations')
plt.savefig('Cost Function over iterations', dpi = 100)
plt.show()
"""




















