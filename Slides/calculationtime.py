import matplotlib.pyplot as plt
import numpy as np
iterations = [1, 2, 3, 4]
circ_no_val = [7, 18, 55, 188] 
circ_val = [10, 38, 183, 790]
lin_no_val = [14, 32, 75, 206]
lin_val =  [13, 35, 96, 370]

plt.rc('text', usetex=True) 
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
plt.rc('text', usetex=True)  # This is Miguel's customization
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
ax.set_xlabel('Iterations', fontsize=18)
ax.set_ylabel('$t_{calc} [s]$', fontsize=18)

ax.set_xticks(np.arange(1, 4.05, 1))
ax.set_yticks(np.arange(0, 801, 100))
ax.set_xlim([1, 4])
ax.set_ylim([0, 800])

plt.plot(iterations, circ_val, label = 'Circular, with validation')
plt.plot(iterations, circ_no_val, label = 'Circular, without validation')
plt.plot(iterations, lin_val, label = 'Linear, with validation')
plt.plot(iterations, lin_no_val, label = 'Linear, without validation')
plt.legend()
plt.title('Calculation time for multiple iterations')
plt.savefig('calculation_time.png', dpi = 100)
plt.show