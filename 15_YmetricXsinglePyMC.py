'''
Estimating the mean and standard deviation of a Gaussian likelihood.
'''
import numpy as np
import pymc3 as pm
from scipy.stats import norm
import matplotlib.pyplot as plt


# THE DATA.

# Generate random data from known parameter values:
np.random.seed(4745)
true_mu = 100
true_std = 15
y = norm.rvs(true_mu, true_std, 500)


# Specify the model in PyMC
with pm.Model() as model:
    # define the priors
    sd = pm.HalfNormal('sd', 25)
    mu = pm.Normal('mu', mu=0, sd=100) # PyMC support precision and std
    #define the likelihood
    yl = pm.Normal('yl', mu, sd, observed=y)
#   Generate a MCMC chain
    trace = pm.sample(5000)


# EXAMINE THE RESULTS

## Print summary for each trace
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace, vars =[mu, tau])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace)

mu_sample = trace['mu']
sigma_sample = trace['sd']



plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 2, 1)
pm.plot_posterior(mu_sample, bins=30, ax=ax)
ax.set_xlabel('mu')
ax.set_title = 'Posterior'
ax.set_xlim(98, 102)

plt.subplot(1, 2, 2)

mu_mean = np.mean(mu_sample)
sigma_mean = np.mean(sigma_sample)

plt.scatter(mu_sample, sigma_sample , c='gray')
plt.plot(mu_mean, sigma_mean, 'C1*',
        label=r'$\mu$ = %.1f, $\sigma$ = %.1f' % (mu_mean, sigma_mean))
plt.xlabel('mu')
plt.ylabel('sigma')
plt.title('Posterior')
plt.legend(loc=0)
plt.savefig('figure_15.3.png')
plt.show()

