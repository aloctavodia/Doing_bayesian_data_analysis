"""
Estimating the mean and standard deviation of a Gaussian likelihood with a
hierarchical model.
"""
from __future__ import division
import numpy as np
import pymc3 as pm
from scipy.stats import norm
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from plot_post import plot_post
from hpd import *
from HtWtDataGenerator import *
import seaborn as sns

# THE DATA.
# Simulated height and weight data:
n_subj = 30
HtWtData = HtWtDataGenerator(n_subj, rndsd=5678)
x = HtWtData[:,1]
y = HtWtData[:,2]

# Re-center data at mean, to reduce autocorrelation in MCMC sampling.
# Standardize (divide by SD) to make initialization easier.
x_m = np.mean(x)
x_sd = np.std(x)
y_m = np.mean(y)
y_sd = np.std(y)
zx = (x - x_m) / x_sd
zy = (y - y_m) / y_sd


# THE MODEL
with pm.Model() as model:
    # define the priors
    tau = pm.Gamma('tau', 0.001, 0.001)
    beta0 = pm.Normal('beta0', mu=0, tau=1.0E-12)
    beta1 = pm.Normal('beta1', mu=0, tau=1.0E-12)
    mu = beta0 + beta1 * zx
    # define the likelihood
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=zy)
    # Generate a MCMC chain
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(10000, step, start, progressbar=False)


# EXAMINE THE RESULTS
burnin = 5000
thin = 10

## Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars =[tau])
#pm.autocorrplot(trace, vars =[tau])

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)

## Extract chain values:
z0 = trace['beta0']
z1 = trace['beta1']
z_tau = trace['tau']
z_sigma = 1 / np.sqrt(z_tau) # Convert precision to SD


# Convert to original scale:
b1 = z1 * y_sd / x_sd
b0 = (z0 * y_sd + y_m - z1 * y_sd * x_m / x_sd)
sigma = z_sigma * y_sd


# Posterior prediction:
# Specify x values for which predicted y's are needed:
x_post_pred = np.arange(55, 81)
# Define matrix for recording posterior predicted y values at each x value.
# One row per x value, with each row holding random predicted y values.
post_samp_size = len(b1)
y_post_pred = np.zeros((len(x_post_pred), post_samp_size))
# Define matrix for recording HDI limits of posterior predicted y values:
y_HDI_lim = np.zeros((len(x_post_pred), 2))
# Generate posterior predicted y values.
# This gets only one y value, at each x, for each step in the chain.
for chain_idx in range(post_samp_size):
    y_post_pred[:,chain_idx] = norm.rvs(loc=b0[chain_idx] + b1[chain_idx] * x_post_pred ,
                           scale = np.repeat([sigma[chain_idx]], [len(x_post_pred)]), size=len(x_post_pred))

for x_idx in range(len(x_post_pred)):
    y_HDI_lim[x_idx] = hpd(y_post_pred[x_idx])

## Display believable beta0 and b1 values
plt.figure()
plt.subplot(1, 2, 1)
thin_idx = 50
plt.plot(z1[::thin_idx], z0[::thin_idx], 'b.', alpha=0.7)
plt.ylabel('Standardized Intercept')
plt.xlabel('Standardized Slope')
plt.subplot(1, 2, 2)
plt.plot(b1[::thin_idx], b0[::thin_idx], 'b.', alpha=0.7)
plt.ylabel('Intercept (ht when wt=0)')
plt.xlabel('Slope (pounds per inch)')
plt.tight_layout()
plt.savefig('Figure_16.4.png')

# Display the posterior of the b1:
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plot_post(z1, xlab='Standardized slope', 
          comp_val=0.0, bins=30, show_mode=False)
plt.subplot(1, 2, 2)
plot_post(b1, xlab='Slope (pounds per inch)', 
          comp_val=0.0, bins=30, show_mode=False)
plt.tight_layout()
plt.savefig('Figure_16.5.png')

# Display data with believable regression lines and posterior predictions.
plt.figure()
# Plot data values:
x_rang = np.max(x) - np.min(x)
y_rang = np.max(y) - np.min(y)
lim_mult = 0.25
x_lim = [np.min(x)-lim_mult*x_rang, np.max(x)+lim_mult*x_rang]
y_lim = [np.min(y)-lim_mult*y_rang, np.max(y)+lim_mult*y_rang]
plt.plot(x, y, 'k.')
plt.title('Data with credible regression lines')
plt.xlabel('X (height in inches)')
plt.ylabel('Y (weight in pounds)')
plt.xlim(x_lim)
plt.ylim(y_lim)
# Superimpose a smattering of believable regression lines:
for i in range(0, len(b0), 100):
    plt.plot(x, b0[i] + b1[i]*x  , c='k', alpha=0.05 )
plt.savefig('Figure_16.2.png')

# Display data with HDIs of posterior predictions.

plt.figure()
# Plot data values:
y_lim = [np.min(y_HDI_lim), np.max(y_HDI_lim)]
plt.plot(x, y, 'k.')
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.xlabel('X (height in inches)')
plt.ylabel('Y (weight in pounds)')
plt.title('Data with 95% HDI & Mean of Posterior Predictions')
# Superimpose posterior predicted 95% HDIs:
y_post_pred_ave = np.average(y_post_pred, axis=1)
#Book version of the HDI representation
#plt.errorbar(x_post_pred,y_post_pred_ave, 
#             yerr=[abs(y_HDI_lim[:,0]-y_post_pred_ave),
#                   abs(y_HDI_lim[:,1]-y_post_pred_ave)], fmt='.')

#Smoothed version of the HDI representation
x_new = np.linspace(x_post_pred.min(), x_post_pred.max(), 200)
y_HDI_lim_smooth = spline(x_post_pred, y_HDI_lim, x_new)
plt.plot(x_post_pred, y_post_pred_ave)
plt.fill_between(x_new, y_HDI_lim_smooth[:,0], y_HDI_lim_smooth[:,1], alpha=0.3)

plt.savefig('Figure_16.6.png')

plt.show()
