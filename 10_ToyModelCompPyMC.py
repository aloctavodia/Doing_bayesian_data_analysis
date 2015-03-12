"""
Comparing models using Hierarchical modelling. Toy Model.
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from plot_post import plot_post

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
    theta0 = 1 / (1 + pm.exp(-nu)) # theta from model index 0
    theta1 = pm.exp(-eta)    # theta from model index 1
    theta = pm.switch(pm.eq(model_index, 0), theta0, theta1)
    # Likelihood
    y = pm.Bernoulli('y', p=theta, observed=y)
    # Sampling
    start = pm.find_MAP()
    step1 = pm.Metropolis(model.vars[1:])
    step2 = pm.ElemwiseCategoricalStep(var=model_index,values=[0,1])
    trace = pm.sample(10000, [step1, step2], start=start, progressbar=False)


# EXAMINE THE RESULTS.
burnin = 1000
thin = 5

## Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars =[nu, eta])
#pm.autocorrplot(trace, vars =[nu, eta])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)


model_idx_sample = trace['model_index'][burnin::thin]
pM1 = sum(model_idx_sample == 0) / len(model_idx_sample)
pM2 = 1 - pM1


nu_sample_M1 = trace['nu'][burnin::thin][model_idx_sample == 0]
eta_sample_M2 = trace['eta'][burnin::thin][model_idx_sample == 1]

plt.figure()
plt.subplot(2, 1, 1)
plot_post(nu_sample_M1, xlab=r'$\nu$', show_mode=False, labelsize=9, framealpha=0.5)
plt.xlabel(r'$\nu$')
plt.ylabel('frequency')
plt.title(r'p($\nu$|D,M2), with p(M2|D)=%.3f' % pM1, fontsize=14)
plt.xlim(-8, 8)

plt.subplot(2, 1, 2)
plot_post(eta_sample_M2, xlab=r'$\eta$', show_mode=False, labelsize=9, framealpha=0.5)
plt.xlabel(r'$\eta$')
plt.ylabel('frequency')
plt.title(r'p($\eta$|D,M2), with p(M2|D)=%.3f' % pM2, fontsize=14)
plt.xlim(0, 8)
plt.savefig('figure_ex_10.2_a.png')
plt.show()
