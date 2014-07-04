"""
Graph of normal probability density function, with comb of intervals.
"""
import matplotlib.pyplot as plt
import numpy as np

meanval = 0.0              # Specify mean of distribution.
sdval = 0.2                # Specify standard deviation of distribution.
xlow = meanval - 3 * sdval  # Specify low end of x-axis.
xhigh = meanval + 3 * sdval  # Specify high end of x-axis.
dx = 0.02                  # Specify interval width on x-axis
# Specify comb points along the x axis:
x = np.arange(xlow, xhigh, dx)
# Compute y values, i.e., probability density at each value of x:
y = (1/(sdval*np.sqrt(2*np.pi))) * np.exp(-.5 * ((x - meanval)/sdval)**2)
# Plot the function. "plot" draws the bell curve. "stem" draws the intervals.
plt.plot(x, y)
plt.stem(x, y, markerfmt=' ')

plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.title('Normal Probability Density')
# Approximate the integral as the sum of width * height for each interval.
area = np.sum(dx*y)
# Display info in the graph.
plt.text(-.6, 1.7, '$\mu$ = %s' % meanval)
plt.text(-.6, 1.5, '$\sigma$ = %s' % sdval)
plt.text(.2, 1.7, '$\Delta x$ = %s' % dx)
plt.text(.2, 1.5, '$\sum_{x}$ $\Delta x$ $p(x)$ = %5.3f' % area)

plt.savefig('Figure_3.3.png')
