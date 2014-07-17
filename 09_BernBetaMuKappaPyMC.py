"""
Bernoulli Likelihood with Hierarchical Prior!
"""
from __future__ import division
import numpy as np
import pymc as pm
import sys
from scipy.stats import beta, gamma
import matplotlib.pyplot as plt
from plot_post import plot_post

np.random.seed(123)

# Data for figure 9.11
N =  [10, 10, 10]  # Number of flips per coin
z =  [5, 5, 5]  # Number of heads per coin

## Data for figure 9.12
#N =  [10, 10, 10]  # Number of flips per coin
#z =  [1, 5, 9]  # Number of heads per coin

# Arrange the data into a more convenient way to feed the PyMC model.
coin = []  # list/vector index for each coins (from 0 to number of coins)
y = []  # list/vector with head (1) or tails (0) for each flip.
for i, flips in enumerate(N):
    heads = z[i]
    if  heads > flips:
        sys.exit("The number of heads can't be greater than the number of flips")
    else:
        y = y + [1] * heads + [0] * (flips-heads)
        coin = coin + [i] * flips

# Specify the model in PyMC
with pm.Model() as model:
# define the hyperparameters
    mu = pm.Beta('mu', 2, 2)
    kappa = pm.Gamma('kappa', 1, 0.1)
    # define the prior
    theta = pm.Beta('theta', mu * kappa, (1 - mu) * kappa, shape=3)
    # define the likelihood
    y = pm.Bernoulli('y', p=theta[coin], observed=y)

    # Generate a MCMC chain
    trace = pm.sample(10000, pm.Metropolis(),
                      progressbar=False)  # Use Metropolis sampling
#    start = pm.find_MAP()  # Find starting value by optimization
#    step = pm.NUTS(state=start)  # Instantiate NUTS sampler
#    trace = pm.sample(20000, step, start=start, progressbar=False)

## Check the results.

## Print summary for each trace
#pm.summary(trace)

## Check for mixing and autocorrelation:
# TODO

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace)


# Create arrays with the posterior sample
burnin = 0  # posterior samples to discard
theta1_sample = trace['theta'][:,0][burnin:]
theta2_sample = trace['theta'][:,1][burnin:]
theta3_sample = trace['theta'][:,2][burnin:]
mu_sample = trace['mu'][burnin:]
kappa_sample = trace['kappa'][burnin:]

fig = plt.figure(figsize=(12,12))

# Scatter plot hyper-parameters
plt.subplot(4, 3, 1)
plt.scatter(mu_sample, kappa_sample, marker='o')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\kappa$')

# Plot mu histogram
plt.subplot(4, 3, 2)
plot_post(mu_sample, xlab=r'$\mu$', show_mode=False, labelsize=9, framealpha=0.5)

# Plot kappa histogram
plt.subplot(4, 3, 3)
plot_post(kappa_sample, xlab=r'$\kappa$', show_mode=False, labelsize=9, framealpha=0.5)

# Plot theta 1
plt.subplot(4, 3, 4)
plot_post(theta1_sample, xlab=r'$\theta1$', show_mode=False, labelsize=9, framealpha=0.5)
plt.xlim(0,1)

# Scatter theta 1 vs mu
plt.subplot(4, 3, 5)
plt.scatter(theta1_sample, mu_sample, marker='o')
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\mu$')

# Scatter theta 1 vs kappa
plt.subplot(4, 3, 6)
plt.scatter(theta1_sample, kappa_sample, marker='o')
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\kappa$')

# Plot theta 2
plt.subplot(4, 3, 7)
plot_post(theta2_sample, xlab=r'$\theta2$', show_mode=False, labelsize=9, framealpha=0.5)
plt.xlim(0,1)

# Scatter theta 2 vs mu
plt.subplot(4, 3, 8)
plt.scatter(theta2_sample, mu_sample, marker='o')
plt.xlabel(r'$\theta2$')
plt.ylabel(r'$\mu$')

# Scatter theta 2 vs kappa
plt.subplot(4, 3, 9)
plt.scatter(theta2_sample, kappa_sample, marker='o')
plt.xlabel(r'$\theta2$')
plt.ylabel(r'$\kappa$')

# Plot theta 3
plt.subplot(4, 3, 10)
plot_post(theta3_sample, xlab=r'$\theta3$', show_mode=False, labelsize=9, framealpha=0.5)
plt.xlim(0,1)

# Scatter theta 3 vs mu
plt.subplot(4, 3, 11)
plt.scatter(theta3_sample, mu_sample, marker='o')
plt.xlabel(r'$\theta3$')
plt.ylabel(r'$\mu$')

# Scatter theta 3 vs kappa
plt.subplot(4, 3, 12)
plt.scatter(theta3_sample, kappa_sample, marker='o')
plt.xlabel(r'$\theta3$')
plt.ylabel(r'$\kappa$')

plt.tight_layout()
plt.savefig('Figure_9.11.png')
plt.show()

