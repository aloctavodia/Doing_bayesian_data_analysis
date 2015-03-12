'''
Estimating the mean and standard deviation of a Gaussian likelihood with a
hierarchical model.
'''
from __future__ import division
import numpy as np
import pymc3 as pm
from scipy.stats import norm
import matplotlib.pyplot as plt
from plot_post import plot_post
import seaborn as sns


# THE DATA.
# Load the aircraft data:
data = np.genfromtxt('Systems.txt', skip_header=True)

n_subj = len(set(data[:,0]))
# Put it into generic variables so easier to change data in other applications:
y = data[:,3]
subj = data[:,0].astype(int)



## Specify the model in PyMC
with pm.Model() as model:
    # define the HyperPriors
    muG = pm.Normal('muG', mu=2.3, tau=0.1)
    tauG = pm.Gamma('tauG', 1, .5)
    m = pm.Gamma('m', 1, .25)
    d = pm.Gamma('d', 1, .5)
    sG = m**2 / d**2
    rG = m / d**2
    # define the priors
    tau = pm.Gamma('tau', sG, rG, shape=n_subj)
    mu = pm.Normal('mu', mu=muG, tau=tauG, shape=n_subj)
    # define the likelihood
    y = pm.Normal('y', mu=mu[subj-1], tau=tau[subj-1], observed=y)
    # Generate a MCMC chain
    #start = pm.find_MAP()
    #step = pm.Metropolis()
    step = pm.Metropolis()
    trace = pm.sample(20000, step, progressbar=False)


# EXAMINE THE RESULTS
burnin = 5000
thin = 100


## Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars =[mu, tau])
#pm.autocorrplot(trace, vars =[mu, tau])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)


## Extract chains
muG_sample = trace['muG'][burnin::thin]
tauG_sample = trace['tauG'][burnin::thin]
m_sample = trace['m'][burnin::thin]
d_sample = trace['d'][burnin::thin]

# Plot the hyperdistributions:
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plot_post(muG_sample, xlab=r'$\mu_g$', bins=30, show_mode=False)
plt.subplot(1, 4, 2)
plot_post(tauG_sample, xlab=r'$\tau_g$', bins=30, show_mode=False)
plt.subplot(1, 4, 3)
plot_post(m_sample, xlab='m', bins=30, show_mode=False)
plt.subplot(1, 4, 4)
plot_post(d_sample, xlab='d', bins=30, show_mode=False)

plt.savefig('Figure_15.9.png')
plt.show()
