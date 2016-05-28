"""
Comparing models using Hierarchical modelling.
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


## specify the Data
y = np.repeat([0, 1], [3, 6])  # 3 tails 6 heads

with pm.Model() as model:
    # Hyperhyperprior:
    model_index = pm.DiscreteUniform('model_index', lower=0, upper=1)
    # Hyperprior:
    kappa_theta = 12
    mu_theta = pm.switch(pm.eq(model_index, 1), 0.25, 0.75)
    # Prior distribution:
    a_theta = mu_theta * kappa_theta
    b_theta = (1 - mu_theta) * kappa_theta
    theta = pm.Beta('theta', a_theta, b_theta) # theta distributed as beta density
    #likelihood
    y = pm.Bernoulli('y', theta, observed=y)
    trace = pm.sample(5000, progressbar=False)


## Check the results.
burnin = 0  # posterior samples to discard

## Print summary for each trace
#pm.df_summary(trace[burnin:])
#pm.df_summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace)

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace[burnin:])
#pm.traceplot(trace)

## Get the posterior sample of model_index:
model_idx_sample = trace['model_index'][burnin:]
## Compute the proportion of model_index at each value:
p_M1 = sum(model_idx_sample == 1) / len(model_idx_sample)
p_M2 = 1 - p_M1


## Get the posterior sample of theta:
theta_sample = trace['theta'][burnin:]
## Extract theta values when model_index is 1:
theta_sample_M1 = theta_sample[model_idx_sample == 1]
## Extract theta values when model_index is 2:
theta_sample_M2 = theta_sample[model_idx_sample == 0]

## Plot histograms of sampled theta values for each model,
plt.figure()
plt.subplot(1, 2, 1)
plt.hist(theta_sample_M1, label='p(M1|D) = %.3f' % p_M1)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|\mu=0.25,D)$')
plt.xlim(0, 1)
plt.legend(loc='upper right', framealpha=0.5)

plt.subplot(1, 2, 2)
plt.hist(theta_sample_M2, label='p(M2|D) = %.3f' % p_M2)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|\mu=0.75,D)$')
plt.xlim(0, 1)
plt.legend(loc='upper right', framealpha=0.5)

plt.savefig('Figure_10.2.png')
plt.show()
