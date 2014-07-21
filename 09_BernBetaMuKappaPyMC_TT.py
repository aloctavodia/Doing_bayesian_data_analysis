"""
Bernoulli Likelihood with Hierarchical Prior. The Therapeutic Touch example.
"""
from __future__ import division
import numpy as np
import pymc as pm
import sys
from scipy.stats import beta, gamma
import matplotlib.pyplot as plt
from plot_post import plot_post

## Therapeutic touch data:
z =  [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
     5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8]  # Number of heads per coin
N =  [10] * len(z)  # Number of flips per coin

# rearrange the data to load it PyMC model.
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
    theta = pm.Beta('theta', mu * kappa, (1 - mu) * kappa, shape=len(N))
    # define the likelihood
    y = pm.Bernoulli('y', p=theta[coin], observed=y)
#   Generate a MCMC chain
    start = pm.find_MAP()  # find a reasonable starting point.
    step1 = pm.Metropolis([theta, mu])
    step2 = pm.NUTS([kappa])
    trace = pm.sample(10000, [step1, step2], start=start, random_seed=(123), progressbar=False)

## Check the results.

## Print summary for each trace
#pm.summary(trace)

## Check for mixing and autocorrelation:
pm.autocorrplot(trace, vars =[mu, kappa])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace)

# Create arrays with the posterior sample
burnin = 2000  # posterior samples to discard
thin = 10  # posterior samples to discard
theta1_sample = trace['theta'][:,0][burnin::thin]
theta28_sample = trace['theta'][:,27][burnin::thin]
mu_sample = trace['mu'][burnin::thin]
kappa_sample = trace['kappa'][burnin::thin]

fig = plt.figure(figsize=(12,12))

# Plot mu histogram
plt.subplot(2, 2, 1)
plot_post(mu_sample, xlab=r'$\mu$', show_mode=False, labelsize=9, framealpha=0.5)

# Plot kappa histogram
plt.subplot(2, 2, 2)
plot_post(kappa_sample, xlab=r'$\kappa$', show_mode=False, labelsize=9, framealpha=0.5)

# Plot theta 1
plt.subplot(2, 2, 3)
plot_post(theta1_sample, xlab=r'$\theta1$', show_mode=False, labelsize=9, framealpha=0.5)

# Plot theta 28
plt.subplot(2, 2, 4)
plot_post(theta28_sample, xlab=r'$\theta28$', show_mode=False, labelsize=9, framealpha=0.5)

plt.tight_layout()
plt.savefig('Figure_9.14.png')
plt.show()
