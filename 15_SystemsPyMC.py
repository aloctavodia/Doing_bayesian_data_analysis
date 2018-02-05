'''
Estimating the mean and standard deviation of a Gaussian likelihood with a
hierarchical model.
'''
from __future__ import division
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

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
    trace = pm.sample(2000)


# EXAMINE THE RESULTS


## Print summary for each trace
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace, vars =[mu, tau])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace)


## Extract chains
muG_sample = trace['muG']
tauG_sample = trace['tauG']
m_sample = trace['m']
d_sample = trace['d']

# Plot the hyperdistributions:
_, ax = plt.subplots(1, 4, figsize=(20, 5))
pm.plot_posterior(muG_sample, bins=30, ax=ax[0])
ax[0].set_xlabel(r'$\mu_g$', fontsize=16)
pm.plot_posterior(tauG_sample, bins=30 ,ax=ax[1])
ax[1].set_xlabel(r'$\tau_g$', fontsize=16)
pm.plot_posterior(m_sample, bins=30, ax=ax[2])
ax[2].set_xlabel('m', fontsize=16)
pm.plot_posterior(d_sample, bins=30, ax=ax[3])
ax[3].set_xlabel('d', fontsize=16)

plt.tight_layout()
plt.savefig('Figure_15.9.png')
plt.show()
