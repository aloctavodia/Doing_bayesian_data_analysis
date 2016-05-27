"""
Bernoulli Likelihood with Hierarchical Prior!
"""
import numpy as np
import pymc3 as pm
import sys
from scipy.stats import beta, binom
import matplotlib.pyplot as plt


# Data for figure 9.11
N =  [10, 10, 10]  # Number of flips per coin
z =  [5, 5, 5]  # Number of heads per coin
## Data for figure 9.12
#N =  [10, 10, 10]  # Number of flips per coin
#z =  [1, 5, 9]  # Number of heads per coin

## Data for exercise 9.1
#ncoins = 50
#nflipspercoin = 5
#mu_act = .7
#kappa_act = 20
#theta_act = beta.rvs(mu_act*kappa_act+1, (1-mu_act)*kappa_act+1, size=ncoins)
#z = binom.rvs(n=nflipspercoin, p=theta_act, size=ncoins)
#N = [nflipspercoin] * ncoins


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
    theta = pm.Beta('theta', mu * kappa, (1 - mu) * kappa, shape=len(N))
    # define the likelihood
    y = pm.Bernoulli('y', p=theta[coin], observed=y)

#   Generate a MCMC chain
    step = pm.Metropolis()
    trace = pm.sample(5000, step, progressbar=False)
#   Restricted models like this could be difficult to sample. This is related 
#   to the censoring comment in the book. One way to detect that something is 
#   wrong with the sampling is to compare the autocorrelation plot and the 
#   sampled values under different sampler, or you can try combinations of 
#   sampler like this

#    step1 = pm.Metropolis([theta, mu])
#    step2 = pm.Slice([kappa])
#    trace = pm.sample(5000, [step1, step2], progressbar=False)

#    or this (this combination was used to generate the figures)

#    start = pm.find_MAP()
#    step1 = pm.Metropolis([theta, mu])
#    step2 = pm.NUTS([kappa])
#    trace = pm.sample(5000, [step1, step2], start=start, progressbar=False)

## Check the results.
burnin = 2000  # posterior samples to discard

## Print summary for each trace
#pm.df_summary(trace[burnin:])
#pm.df_summary(trace)

## Check for mixing and autocorrelation
pm.autocorrplot(trace[burnin:], varnames=['mu', 'kappa'])
#pm.autocorrplot(trace, varnames =[mu, kappa])

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace[burnin:])
#pm.traceplot(trace)

# Create arrays with the posterior sample
theta1_sample = trace['theta'][:,0][burnin:]
theta2_sample = trace['theta'][:,1][burnin:]
theta3_sample = trace['theta'][:,2][burnin:]
mu_sample = trace['mu'][burnin:]
kappa_sample = trace['kappa'][burnin:]


# Scatter plot hyper-parameters
fig, ax = plt.subplots(4, 3, figsize=(12,12))
ax[0, 0].scatter(mu_sample, kappa_sample, marker='o', color='skyblue')
ax[0, 0].set_xlim(0,1)
ax[0, 0].set_xlabel(r'$\mu$')
ax[0, 0].set_ylabel(r'$\kappa$')

# Plot mu histogram
#plot_post(mu_sample, xlab=r'$\mu$', show_mode=False, labelsize=9, framealpha=0.5)

pm.plot_posterior(mu_sample, ax=ax[0, 1], color='skyblue')
ax[0, 1].set_xlabel(r'$\mu$')
ax[0, 1].set_xlim(0,1)

# Plot kappa histogram
#plot_post(kappa_sample, xlab=r'$\kappa$', show_mode=False, labelsize=9, framealpha=0.5)
pm.plot_posterior(kappa_sample, ax=ax[0, 2], color='skyblue')
ax[0, 2].set_xlabel(r'$\kappa$')

# Plot theta 1

#plot_post(theta1_sample, xlab=r'$\theta1$', show_mode=False, labelsize=9, framealpha=0.5)
pm.plot_posterior(theta1_sample, ax=ax[1, 0], color='skyblue')
ax[1, 0].set_xlabel(r'$\theta1$')
ax[1, 0].set_xlim(0,1)

# Scatter theta 1 vs mu
ax[1, 1].scatter(theta1_sample, mu_sample, marker='o', color='skyblue')
ax[1, 1].set_xlim(0,1)
ax[1, 1].set_ylim(0,1)
ax[1, 1].set_xlabel(r'$\theta1$')
ax[1, 1].set_ylabel(r'$\mu$')

# Scatter theta 1 vs kappa
ax[1, 2].scatter(theta1_sample, kappa_sample, marker='o', color='skyblue')
ax[1, 2].set_xlim(0,1)
ax[1, 2].set_xlabel(r'$\theta1$')
ax[1, 2].set_ylabel(r'$\kappa$')

# Plot theta 2
#plot_post(theta2_sample, xlab=r'$\theta2$', show_mode=False, labelsize=9, framealpha=0.5)
pm.plot_posterior(theta2_sample, ax=ax[2, 0], color='skyblue')
ax[2, 0].set_xlabel(r'$\theta2$')
ax[2, 0].set_xlim(0,1)

# Scatter theta 2 vs mu
ax[2, 1].scatter(theta2_sample, mu_sample, marker='o', color='skyblue')
ax[2, 1].set_xlim(0,1)
ax[2, 1].set_ylim(0,1)
ax[2, 1].set_xlabel(r'$\theta2$')
ax[2, 1].set_ylabel(r'$\mu$')

# Scatter theta 2 vs kappa
ax[2, 2].scatter(theta2_sample, kappa_sample, marker='o', color='skyblue')
ax[2, 2].set_xlim(0,1)
ax[2, 2].set_xlabel(r'$\theta2$')
ax[2, 2].set_ylabel(r'$\kappa$')

# Plot theta 3

#plot_post(theta3_sample, xlab=r'$\theta3$', show_mode=False, labelsize=9, framealpha=0.5)
pm.plot_posterior(theta3_sample, ax=ax[3, 0], color='skyblue')
ax[3, 0].set_xlabel(r'$\theta3$')
ax[3, 0].set_xlim(0,1)

# Scatter theta 3 vs mu
ax[3, 1].scatter(theta3_sample, mu_sample, marker='o', color='skyblue')
ax[3, 1].set_xlim(0,1)
ax[3, 1].set_ylim(0,1)
ax[3, 1].set_xlabel(r'$\theta3$')
ax[3, 1].set_ylabel(r'$\mu$')

# Scatter theta 3 vs kappa
ax[3, 2].scatter(theta3_sample, kappa_sample, marker='o', color='skyblue')
ax[3, 2].set_xlim(0,1)
ax[3, 2].set_xlabel(r'$\theta3$')
ax[3, 2].set_ylabel(r'$\kappa$')

plt.tight_layout()
plt.savefig('Figure_9.11.png')
plt.show()

