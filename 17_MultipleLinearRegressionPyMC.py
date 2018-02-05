"""
Multiple linear regression
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from hpd import *
import seaborn as sns


# THE DATA.
dataSource = ["Guber1999", "McIntyre1994", "random"][0]

if dataSource == "Guber1999":
    fname = "Guber1999" # file name for saved graphs
    data = pd.read_csv('Guber1999data.txt', sep='\s+', 
    names = ["State","Spend","StuTchRat","Salary", "PrcntTake","SATV","SATM","SATT"])
    # Specify variables to be used in BUGS analysis:
    predictedName = "SATT"
    predictorNames = ["Spend" , "PrcntTake"]
    nData = len(data)
    y = data[predictedName]
    x = data[predictorNames]
    n_predictors = len(x.columns)


if dataSource == "McIntyre1994":
    fname = "McIntyre1994" # file name for saved graphs
    data = pd.read_csv('McIntyre1994data.csv')
    predictedName = "CO"
    predictorNames = ["Tar","Nic","Wt"]
    nData = len(data)
    y = data[predictedName]
    x = data[predictorNames]
    nData = len(data)


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
    nData = 100
    x = norm.rvs(xM, xSD, n_predictors*nData).reshape(100, -1)
    x = pd.DataFrame(x, columns=['X%s' % i for i in range(0, n_predictors)])
    # Random Y values generated from linear model with true parameter values:
    y = np.sum(x * beta_true[1:].T, axis=1) + beta_true[0] + norm.rvs(0, sd_true, nData)
    # Select which predictors to include
    includeOnly = range(0, n_predictors) # default is to include all
    #x = x.iloc[includeOnly]
    predictorNames = x.columns
    n_predictors = len(predictorNames)



# THE MODEL
with pm.Model() as model:
    # define the priors
    beta0 = pm.Normal('beta0', mu=0, sd=100)
    beta1 = pm.Normal('beta1', mu= 0, sd=100, shape=n_predictors)
    sd = pm.HalfNormal('sd', 25)
    mu = beta0 + pm.math.dot(beta1, x.values.T)
    # define the likelihood
    yl = pm.Normal('yl', mu, sd, observed=y)
    # Generate a MCMC chain
    trace = pm.sample(1000)

# EXAMINE THE RESULTS

# Print summary for each trace
#pm.summary(trace)

# Check for mixing and autocorrelation
#pm.autocorrplot(trace, vars =[beta0])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace)


# Extract chain values:
b0_samp = trace['beta0']
b_samp = trace['beta1']

Sigma_samp = trace['sd']
chain_length = len(Sigma_samp)

if n_predictors >= 6: # don't display if too many predictors
    n_predictors == 6

columns = ['Sigma y', 'Intercept']
[columns.append('Slope_%s' % i) for i in predictorNames[:n_predictors]]
traces = np.array([Sigma_samp, b0_samp, b_samp[:,0], b_samp[:,1]]).T
df = pd.DataFrame(traces, columns=columns)
sns.set_style('dark')
g = sns.PairGrid(df)
g.map(plt.scatter)
plt.savefig('Figure_17.5b.png')

## Display the posterior:
sns.set_style('darkgrid')

plt.figure(figsize=(16,4))
ax = plt.subplot(1, n_predictors+2, 1)
pm.plot_posterior(Sigma_samp, ax=ax)
ax.set_xlabel(r'$\sigma y$') 
ax = plt.subplot(1, n_predictors+2, 2)
ax = pm.plot_posterior(b0_samp,  ax=ax)
ax.set_xlabel('Intercept')

for i in range(0, n_predictors):
    ax = plt.subplot(1, n_predictors+2, 3+i)
    pm.plot_posterior(b_samp[:,i], ref_val=0, ax=ax)
    ax.set_xlabel('Slope_{}'.format(predictorNames[i]))
plt.tight_layout()
plt.savefig('Figure_17.5a.png')


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
                                        scale = np.repeat([Sigma_samp[chain_idx]], [len(x)]))

for x_idx in range(len(x)):
    y_HDI_lim[x_idx] = hpd(y_post_pred[x_idx])

for i in range(len(x)):
    print(np.mean(y_post_pred, axis=1)[i], y_HDI_lim[i])

plt.show()
