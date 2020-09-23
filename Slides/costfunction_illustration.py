import matplotlib.pyplot as plt
from numpy import arange

def parabola(x):
    return (x-3)**2


fig, ax = plt.subplots(figsize=(5, 5))
x = arange(0, 6.2, 0.2)
plt.plot(x, parabola(x), label = 'Cost Function')
circle1 = plt.Circle((5, parabola(5)), 0.1, color='black', label = 'Initial Guess')
circle2 = plt.Circle((3, 0), 0.1, color='red', label = 'Global Optimum')
ax.add_artist(circle1)
ax.add_artist(circle2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_ylim([0, 9])
ax.set_xlim([0, 9])
ax.legend([circle1, circle2], ['Initial Guess', 'Global Optimum'])
ax.set_xlabel('a', fontsize=18)
ax.set_ylabel('f(a)', fontsize=18)
plt.title('Sketch of a Cost Function for a 1d Problem')
plt.savefig('Cost Function', dpi = 100)
plt.show()