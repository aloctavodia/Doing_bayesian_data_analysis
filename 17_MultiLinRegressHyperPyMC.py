"""
Multiple linear regression with hyperpriors.
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from plot_post import plot_post
from hpd import *
import seaborn as sns


# THE DATA.

tdfBgain = 1

dataSource = ["Guber1999", "McIntyre1994", "random"][0]

if dataSource == "Guber1999":
    fname = "Guber1999" # file name for saved graphs
    data = pd.read_csv('Guber1999data.txt', sep='\s+', 
    names = ["State","Spend","StuTchRat","Salary", "PrcntTake","SATV","SATM","SATT"])
    # Specify variables to be used in BUGS analysis:
    predicted_name = "SATT"
    predictor_names = ["Spend" , "PrcntTake"]
    n_data = len(data)
    y = data[predicted_name]
    x = data[predictor_names]
    n_predictors = len(x.columns)


if dataSource == "McIntyre1994":
    fname = "McIntyre1994" # file name for saved graphs
    data = pd.read_csv('McIntyre1994data.csv')
    predicted_name = "CO"
    predictor_names = ["Tar","Nic","Wt"]
    n_data = len(data)
    y = data[predicted_name]
    x = data[predictor_names]
    n_data = len(data)


if dataSource == "random":
    fname = "Random"  # file name for saved graphs
    # Generate random data.
    # True parameter values:
    beta_true = np.repeat(0, 21)
    beta_true = np.insert(beta_true, [0,0,0], [100, 1, 2])  # beta0 is first component
    n_predictors = len(beta_true) - 1
    sd_true = 2
    tau_true = 1/sd_true**2
    # Random X values:
    np.random.seed(47405)
    xM = 5
    xSD = 2
    n_data = 100
    x = norm.rvs(xM, xSD, n_predictors*n_data).reshape(100, -1)
    x = pd.DataFrame(x, columns=['X%s' % i for i in range(0, n_predictors)])
    # Random Y values generated from linear model with true parameter values:
    y = np.sum(x * beta_true[1:].T, axis=1) + beta_true[0] + norm.rvs(0, sd_true, n_data)
   # Select which predictors to include
    include_only = range(0, n_predictors) # default is to include all
    #x = x.iloc[include_only]
    predictor_names = x.columns
    n_predictors = len(predictor_names)


# THE MODEL
with pm.Model() as model:
    # define hyperpriors
    muB = pm.Normal('muB', 0,.100 )
    tauB = pm.Gamma('tauB', .01, .01)
    udfB = pm.Uniform('udfB', 0, 1)
    tdfB = 1 + tdfBgain * (-pm.log(1 - udfB))
    # define the priors
    tau = pm.Gamma('tau', 0.01, 0.01)
    beta0 = pm.Normal('beta0', mu=0, tau=1.0E-12)
    beta1 = pm.T('beta1', mu=muB, lam=tauB, nu=tdfB, shape=n_predictors)
    mu = beta0 + pm.dot(beta1, x.values.T)
    # define the likelihood
    #mu = beta0 + beta1[0] * x.values[:,0] + beta1[1] * x.values[:,1]
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=y)
    # Generate a MCMC chain
    start = pm.find_MAP()
    step1 = pm.NUTS([beta1])
    step2 = pm.Metropolis([beta0, tau, muB, tauB, udfB])
    trace = pm.sample(10000, [step1, step2], start, progressbar=False)


# EXAMINE THE RESULTS
burnin = 2000
thin = 1

# Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

# Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars =[mu, tau])
#pm.autocorrplot(trace, vars =[beta0])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)


# Extract chain values:
b0_samp = trace['beta0'][burnin::thin]
b_samp = trace['beta1'][burnin::thin]
tau_samp = trace['tau'][burnin::thin]
sigma_samp = 1 / np.sqrt(tau_samp) # Convert precision to SD
chain_length = len(tau_samp)

if n_predictors >= 6: # don't display if too many predictors
    n_predictors == 6

columns = ['Sigma y', 'Intercept']
[columns.append('Slope_%s' % i) for i in predictor_names[:n_predictors]]
traces = np.array([sigma_samp, b0_samp, b_samp[:,0], b_samp[:,1]]).T
df = pd.DataFrame(traces, columns=columns)
sns.set_style('dark')
g = sns.PairGrid(df)
g.map(plt.scatter)
plt.savefig('Figure_17.Xa.png')

## Display the posterior:
sns.set_style('darkgrid')

plt.figure(figsize=(16,4))
plt.subplot(1, n_predictors+2, 1)
plot_post(sigma_samp, xlab=r'$\sigma y$', show_mode=False, framealpha=0.5)
plt.subplot(1, n_predictors+2, 2)
plot_post(b0_samp, xlab='Intercept', show_mode=False, framealpha=0.5)

for i in range(0, n_predictors):
    plt.subplot(1, n_predictors+2, 3+i)
    plot_post(b_samp[:,i], xlab='Slope_%s' % predictor_names[i],
              show_mode=False, framealpha=0.5, comp_val=0)
plt.tight_layout()
plt.savefig('Figure_17.Xb.png')

# Posterior prediction:
# Define matrix for recording posterior predicted y values for each xPostPred.
# One row per xPostPred value, with each row holding random predicted y values.
y_post_pred = np.zeros((len(x), chain_length))
# Define matrix for recording HDI limits of posterior predicted y values:
y_HDI_lim = np.zeros((len(x), 2))
# Generate posterior predicted y values.
# This gets only one y value, at each x, for each step in the chain.
#or chain_idx in range(chain_length):
for chain_idx in range(chain_length):
    y_post_pred[:,chain_idx] = norm.rvs(loc = b0_samp[chain_idx] + np.dot(b_samp[chain_idx], x.values.T), 
                                        scale = np.repeat([sigma_samp[chain_idx]], [len(x)]))

for x_idx in range(len(x)):
    y_HDI_lim[x_idx] = hpd(y_post_pred[x_idx])

for i in range(len(x)):
    print np.mean(y_post_pred, axis=1)[i], y_HDI_lim[i]

plt.show()
