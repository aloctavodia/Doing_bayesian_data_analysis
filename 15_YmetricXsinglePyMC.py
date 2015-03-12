'''
Estimating the mean and standard deviation of a Gaussian likelihood.
'''
import numpy as np
import pymc3 as pm
from scipy.stats import norm
import matplotlib.pyplot as plt
from plot_post import plot_post
import seaborn as sns

# THE DATA.

# Generate random data from known parameter values:
np.random.seed(4745)
true_mu = 100
true_std = 15
y = norm.rvs(true_mu, true_std, 500)


# Specify the model in PyMC
with pm.Model() as model:
    # define the priors
    tau = pm.Gamma('tau', 0.01, 0.01)
    mu = pm.Normal('mu', mu=0, tau=1E-10) # PyMC support precission and std
    #define the likelihood
    y = pm.Normal('y', mu=mu, tau=tau, observed=y)
#   Generate a MCMC chain
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(5000, step, start, progressbar=False)


# EXAMINE THE RESULTS
burnin = 1000
thin = 10


## Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars =[mu, tau])
#pm.autocorrplot(trace, vars =[mu, tau])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)

mu_sample = trace['mu'][burnin::thin]
tau_sample = trace['tau'][burnin::thin]
sigma_sample = 1 / np.sqrt(tau_sample) # Convert precision to std


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plot_post(mu_sample, xlab='mu', bins=30, title='Posterior', show_mode=False)
plt.xlim(98, 102)

plt.subplot(1, 2, 2)

mu_mean = np.mean(mu_sample)
sigma_mean = np.mean(sigma_sample)

plt.scatter(mu_sample, sigma_sample , c='gray')
plt.plot(mu_mean, sigma_mean, 'r*',
        label=r'$\mu$ = %.1f, $\sigma$ = %.1f' % (mu_mean, sigma_mean))
plt.xlabel('mu')
plt.ylabel('sigma')
plt.title('Posterior')
plt.legend(loc=0)
plt.savefig('figure_15.3.png')
plt.show()

