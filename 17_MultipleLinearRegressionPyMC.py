"""
Multiple linear regression
"""
from __future__ import division
import numpy as np
import pymc as pm
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from plot_post import plot_post
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
    nPredictors = len(x.columns)


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
    betaTrue = np.repeat(0, 21)
    betaTrue = np.insert(betaTrue, [0,0,0], [100, 1, 2])  # beta0 is first component
    nPredictors = len(betaTrue) - 1
    sdTrue = 2
    tauTrue = 1/sdTrue**2
    # Random X values:
    np.random.seed(47405)
    xM = 5
    xSD = 2
    nData = 100
    x = norm.rvs(xM, xSD, nPredictors*nData).reshape(100, -1)
    x = pd.DataFrame(x, columns=['X%s' % i for i in range(0, nPredictors)])
    # Random Y values generated from linear model with true parameter values:
    y = np.sum(x * betaTrue[1:].T, axis=1) + betaTrue[0] + norm.rvs(0, sdTrue, nData)
    predictedName = "Y"
   # Select which predictors to include
    includeOnly = range(0, nPredictors) # default is to include all
    x = x.iloc[includeOnly]
    predictorNames = x.columns
    nPredictors = len(predictorNames)


# THE MODEL
with pm.Model() as model:
    # define the priors
    beta0 = pm.Normal('beta0', mu=0, tau=1.0E-12)
    beta1 = pm.Normal('beta1', mu= 0, tau=1.0E-12, shape=nPredictors)
    tau = pm.Gamma('tau', 0.01, 0.01)
    # define the likelihood
    #mu = beta0 + beta1[0] * x.values[:,0] + beta1[1] * x.values[:,1]
    mu = beta0 + pm.dot(beta1, x.values.T)
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=y)
    # Generate a MCMC chain
    start = pm.find_MAP()
    step1 = pm.NUTS([beta1])
    step2 = pm.Metropolis([beta0, tau])
    trace = pm.sample(10000, [step1, step2], start, progressbar=False)

# EXAMINE THE RESULTS
burnin = 5000
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
b0Samp = trace['beta0'][burnin::thin]
bSamp = trace['beta1'][burnin::thin]
TauSamp = trace['tau'][burnin::thin]
SigmaSamp = 1 / np.sqrt(TauSamp) # Convert precision to SD
chainLength = len(TauSamp)

if nPredictors >= 6: # don't display if too many predictors
    nPredictors == 6

columns = ['Sigma y', 'Intercept']
[columns.append('Slope_%s' % i) for i in predictorNames[:nPredictors]]
traces = np.array([SigmaSamp, b0Samp, bSamp[:,0], bSamp[:,1]]).T ## XXX
df = pd.DataFrame(traces, columns=columns)
plt.figure()
sns.set_style('dark')
g = sns.PairGrid(df)
g.map(plt.scatter)
plt.savefig('Figure_17.5b.png')

## Display the posterior:
sns.set_style('darkgrid')

plt.figure(figsize=(16,4))
plt.subplot(1, nPredictors+2, 1)
plot_post(SigmaSamp, xlab=r'$\sigma y$', show_mode=False, framealpha=0.5)
plt.subplot(1, nPredictors+2, 2)
plot_post(b0Samp, xlab='Intercept', show_mode=False, framealpha=0.5)

for i in range(0, nPredictors):
    plt.subplot(1, nPredictors+2, 3+i)
    plot_post(bSamp[:,i], xlab='Slope_%s' % predictorNames[i],
              show_mode=False, framealpha=0.5, comp_val=0)
plt.tight_layout()
plt.savefig('Figure_17.5a.png')


plt.show()
