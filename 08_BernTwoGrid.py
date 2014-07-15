"""
Inferring two binomial proportions via grid aproximation.
"""
from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import beta
from HDI_of_grid import HDI_of_grid
import numpy as np


# Specify the grid on theta1,theta2 parameter space.
n_int = 500  # arbitrary number of intervals for grid on theta.
theta1 = np.linspace(0, 1, n_int)
theta2 = theta1

theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)

# Specify the prior probability _masses_ on the grid.
prior_name = ("Beta","Ripples","Null","Alt")[0]  # or define your own.
if prior_name == "Beta":
    a1, b1, a2, b2 = 3, 3, 3, 3
    prior1 = beta.pdf(theta1_grid, a1, b1)
    prior2 = beta.pdf(theta2_grid, a1, b1)
    prior = prior1 * prior2
    prior = prior / np.sum(prior)

if prior_name == "Ripples":
    m1, m2, k = 0, 1, 0.75 * np.pi
    prior = np.sin((k*(theta1_grid-m1))**2 + (k*(theta2_grid-m2))**2)**2
    prior = prior / np.sum(prior)

if prior_name == "Null":
    # 1's at theta1=theta2, 0's everywhere else:
    prior = np.eye(len(theta1_grid), len(theta2_grid))
    prior = prior / np.sum(prior)

if prior_name == "Alt":
#    # Uniform:
    prior = np.ones((len(theta1_grid), len(theta2_grid)))
    prior = prior / np.sum(prior)

# Specify likelihood
z1, N1, z2, N2 = 5, 7, 2, 7  # data are specified here
likelihood = theta1_grid**z1 * (1-theta1_grid)**(N1-z1) * theta2_grid**z2 * (1-theta2_grid)**(N2-z2)

# Compute posterior from point-by-point multiplication and normalizing:
p_data = np.sum(prior * likelihood)
posterior = (prior * likelihood) / p_data

# Specify the probability mass for the HDI region
credib = .95
thin = 4
color = 'skyblue'

fig = plt.figure(figsize=(12,12))

# prior
ax = fig.add_subplot(3, 2, 1, projection='3d')
ax.plot_surface(theta1_grid[::thin,::thin], theta2_grid[::thin,::thin], prior[::thin,::thin], color=color)
ax.set_xlabel(r'$\theta1$')
ax.set_ylabel(r'$\theta1$')
ax.set_zlabel(r'$p(t1,t2)$')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.subplot(3, 2, 2)
plt.contour(theta1_grid, theta2_grid, prior, colors=color)
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\theta1$')

# likelihood
ax = fig.add_subplot(3, 2, 3, projection='3d')
ax.plot_surface(theta1_grid[::thin,::thin], theta2_grid[::thin,::thin], likelihood[::thin,::thin], color=color)
ax.set_xlabel(r'$\theta1$')
ax.set_ylabel(r'$\theta1$')
ax.set_zlabel(r'$p(D|t1,t2)$')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.subplot(3, 2, 4)
plt.contour(theta1_grid, theta2_grid, likelihood, colors=color)
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\theta1$')
plt.plot(0, label='z1,N1,z2,N2=%s,%s,%s,%s' % (z1, N1, z2, N2), alpha=0)
plt.legend(loc='upper left')

# posterior
ax = fig.add_subplot(3, 2, 5, projection='3d')
ax.plot_surface(theta1_grid[::thin,::thin], theta2_grid[::thin,::thin],posterior[::thin,::thin], color=color)
ax.set_xlabel(r'$\theta1$')
ax.set_ylabel(r'$\theta1$')
ax.set_zlabel(r'$p(t1,t2|D)$')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.subplot(3, 2, 6)
plt.contour(theta1_grid, theta2_grid, posterior, colors=color)
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\theta1$')
plt.plot(0, label='p(D) = %.3e' % p_data, alpha=0)
plt.legend(loc='upper left')

# Mark the highest posterior density region
HDI_height = HDI_of_grid(posterior)['height']
plt.contour(theta1_grid, theta2_grid, posterior, levels=[HDI_height], colors='k')

plt.tight_layout()
plt.savefig('BernTwoGrid_%s.png' % prior_name)
plt.show()
