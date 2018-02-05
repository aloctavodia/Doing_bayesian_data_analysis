"""
Comparing models using Hierarchical modelling. Toy Model.
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# THE DATA.
N = 30
z = 8
y = np.repeat([1, 0], [z, N-z]) 

# THE MODEL.
with pm.Model() as model:
    # Hyperprior on model index:
    model_index = pm.DiscreteUniform('model_index', lower=0, upper=1)
    # Prior
    nu = pm.Normal('nu', mu=0, tau=0.1) # it is posible to use tau or sd
    eta = pm.Gamma('eta', .1, .1)
    theta0 = 1 / (1 + pm.math.exp(-nu)) # theta from model index 0
    theta1 = pm.math.exp(-eta)    # theta from model index 1
    theta = pm.math.switch(pm.math.eq(model_index, 0), theta0, theta1)
    # Likelihood
    y = pm.Bernoulli('y', p=theta, observed=y)
    # Sampling
    trace = pm.sample(1000)


# EXAMINE THE RESULTS.
## Print summary for each trace

#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace, vars =[nu, eta])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace)


model_idx_sample = trace['model_index']
pM1 = sum(model_idx_sample == 0) / len(model_idx_sample)
pM2 = 1 - pM1


nu_sample_M1 = trace['nu'][model_idx_sample == 0]
eta_sample_M2 = trace['eta'][model_idx_sample == 1]

plt.figure()
plt.subplot(2, 1, 1)
pm.plot_posterior(nu_sample_M1)
plt.xlabel(r'$\nu$')
plt.ylabel('frequency')
plt.title(r'p($\nu$|D,M2), with p(M2|D)={:.3}f'.format(pM1), fontsize=14)
plt.xlim(-8, 8)

plt.subplot(2, 1, 2)
pm.plot_posterior(eta_sample_M2)
plt.xlabel(r'$\eta$')
plt.ylabel('frequency')
plt.title(r'p($\eta$|D,M2), with p(M2|D)={:.3f}'.format(pM2), fontsize=14)
plt.xlim(0, 8)
plt.savefig('figure_ex_10.2_a.png')
plt.show()
