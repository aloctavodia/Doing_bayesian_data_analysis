"""
More Hierarchical models. The filtration-condensation experiment. 
Hyperparameters for kappafor all conditions.
"""
import numpy as np
import pymc as pm
import sys
import matplotlib.pyplot as plt
from plot_post import plot_post


# Data
# For each subject, specify the condition s/he was in,
# the number of trials s/he experienced, and the number correct.
ncond = 4
nSubj = 40
trials = 64

N = np.repeat([trials], (ncond * nSubj))
z = np.array([45, 63, 58, 64, 58, 63, 51, 60, 59, 47, 63, 61, 60, 51, 59, 45,
61, 59, 60, 58, 63, 56, 63, 64, 64, 60, 64, 62, 49, 64, 64, 58, 64, 52, 64, 64,
64, 62, 64, 61, 59, 59, 55, 62, 51, 58, 55, 54, 59, 57, 58, 60, 54, 42, 59, 57,
59, 53, 53, 42, 59, 57, 29, 36, 51, 64, 60, 54, 54, 38, 61, 60, 61, 60, 62, 55,
38, 43, 58, 60, 44, 44, 32, 56, 43, 36, 38, 48, 32, 40, 40, 34, 45, 42, 41, 32,
48, 36, 29, 37, 53, 55, 50, 47, 46, 44, 50, 56, 58, 42, 58, 54, 57, 54, 51, 49,
52, 51, 49, 51, 46, 46, 42, 49, 46, 56, 42, 53, 55, 51, 55, 49, 53, 55, 40, 46,
56, 47, 54, 54, 42, 34, 35, 41, 48, 46, 39, 55, 30, 49, 27, 51, 41, 36, 45, 41,
53, 32, 43, 33])
condition = np.repeat([0,1,2,3], nSubj)

# Specify the model in PyMC
with pm.Model() as model:
    # define the hyper-hyperparameters for kappa
    mean_gamma = pm.Uniform('mean_gamma', 0, 30)
    sd_gamma = pm.Uniform('sd_gamma', 0, 30)
    s_kappa = mean_gamma**2/sd_gamma**2
    r_kappa = mean_gamma/sd_gamma**2
    # define the hyperparameters
    kappa = pm.Gamma('kappa', s_kappa, r_kappa)
    mu = pm.Beta('mu', 1, 1, shape=ncond)
    # define the prior
    theta = pm.Beta('theta', mu[condition] * kappa, (1 - mu[condition]) * kappa, shape=len(z))
    # define the likelihood
    y = pm.Binomial('y', p=theta, n=N, observed=z)
    start = pm.find_MAP()
    step1 = pm.Metropolis([mu])
    step2 = pm.Metropolis([theta])
    step3 = pm.NUTS([kappa])
#    samplers = [pm.Metropolis([rv]) for rv in model.unobserved_RVs]
    trace = pm.sample(10000, [step1, step2, step3], start=start, progressbar=False)

## Check the results.
burnin = 5000  # posterior samples to discard
thin = 10  # posterior samples to discard

## Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace, vars =[mu, kappa])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
pm.traceplot(trace)


# Create arrays with the posterior sample
mu1_sample = trace['mu'][:,0][burnin::thin]
mu2_sample = trace['mu'][:,1][burnin::thin]
mu3_sample = trace['mu'][:,2][burnin::thin]
mu4_sample = trace['mu'][:,3][burnin::thin]


fig = plt.figure(figsize=(15, 6))

# Plot differences among filtrations experiments
plt.subplot(1, 3, 1)
plot_post((mu1_sample-mu2_sample), xlab=r'$\mu1-\mu2$', show_mode=False, comp_val=0, framealpha=0.5)

# Plot differences among condensation experiments
plt.subplot(1, 3, 2)
plot_post((mu3_sample-mu4_sample), xlab=r'$\mu3-\mu4$', show_mode=False, comp_val=0, framealpha=0.5)

# Plot differences between filtration and condensation experiments
plt.subplot(1, 3, 3)
a = (mu1_sample+mu2_sample)/2 - (mu3_sample+mu4_sample)/2
plot_post(a, xlab=r'$(\mu1+\mu2)/2 - (\mu3+\mu4)/2$', show_mode=False, comp_val=0, framealpha=0.5)

plt.tight_layout()
plt.savefig('Figure_9.18_lower.png')
plt.show()
