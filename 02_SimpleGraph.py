"""
A simple graph drawn by Python :-)
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

x = np.linspace(-2, 2, 40)
y = x**2

plt.plot(x, y)
plt.savefig('Figure_2.2.png')
plt.show()
