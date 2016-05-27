"""
Bernoulli Likelihood with Hierarchical Prior. The Therapeutic Touch example.
"""
import numpy as np
import pymc3 as pm
import sys
import matplotlib.pyplot as plt


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
    trace = pm.sample(10000, step=pm.NUTS(), random_seed=(123), progressbar=False)

## Check the results.
burnin = 100  # posterior samples to discard

## Print summary for each trace
#pm.df_summary(trace[burnin:])
#pm.df_summary(trace)

## Check for mixing and autocorrelation
pm.autocorrplot(trace[burnin:], varnames=['mu', 'kappa'])
#pm.autocorrplot(trace, vars =[mu, kappa])

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace[burnin:])
#pm.traceplot(trace)

# Create arrays with the posterior sample
theta1_sample = trace['theta'][:,0][burnin:]
theta28_sample = trace['theta'][:,27][burnin:]
mu_sample = trace['mu'][burnin:]
kappa_sample = trace['kappa'][burnin:]

# Plot mu histogram
fig, ax = plt.subplots(2, 2, figsize=(12,12))
pm.plot_posterior(mu_sample, ax=ax[0, 0], color='skyblue')
ax[0, 0].set_xlabel(r'$\mu$')

# Plot kappa histogram
pm.plot_posterior(kappa_sample, ax=ax[0, 1], color='skyblue')
ax[0, 1].set_xlabel(r'$\kappa$')

# Plot theta 1
pm.plot_posterior(theta1_sample, ax=ax[1, 0], color='skyblue')
ax[1, 0].set_xlabel(r'$\theta1$')

# Plot theta 28
pm.plot_posterior(theta1_sample, ax=ax[1, 1], color='skyblue')
ax[1, 1].set_xlabel(r'$\theta28$')


plt.tight_layout()
plt.savefig('Figure_9.14.png')
plt.show()
