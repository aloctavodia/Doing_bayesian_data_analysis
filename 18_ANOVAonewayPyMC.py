"""
One way BANOVA
"""
from __future__ import division
import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
from plot_post import plot_post
from hpd import * 
import seaborn as sns

# THE DATA.
# Specify data source:
dataSource = ["McDonaldSK1991" , "SolariLS2008" , "Random"][0]

# Load the data:
if dataSource == "McDonaldSK1991":
    datarecord = pd.read_csv("McDonaldSK1991data.txt", sep='\s+', skiprows=18, skipfooter=25)
    y = datarecord['Size']
    Ntotal = len(y)
    x = datarecord['Group'] - 1
    xnames = pd.unique(datarecord['Site'])
    NxLvl = len(xnames)
    contrast_dict = {'BIGvSMALL':[-1/3,-1/3,1/2,-1/3,1/2],
                       'ORE1vORE2': [1,-1,0,0,0],
                       'ALAvORE':[-1/2,-1/2,1,0,0],
                       'NPACvORE':[-1/2,-1/2,1/2,1/2,0],
                       'USAvRUS':[1/3,1/3,1/3,-1,0],
                       'FINvPAC':[-1/4,-1/4,-1/4,-1/4,1],
                       'ENGvOTH':[1/3,1/3,1/3,-1/2,-1/2],
                       'FINvRUS':[0,0,0,-1,1]}

z = (y - y.mean())/y.std()

## THE MODEL.
with pm.Model() as model:
    # define the hyperpriors
    a_SD_unabs = pm.T('a_SD_unabs', mu=0, lam=0.001, nu=1)
    a_SD = abs(a_SD_unabs) + 0.1
    atau = 1 / a_SD**2
    # define the priors
    sigma = pm.Uniform('sigma', 0, 10) # y values are assumed to be standardized
    tau = 1 / sigma**2
    a0 = pm.Normal('a0', mu=0, tau=0.001) # y values are assumed to be standardized
    a = pm.Normal('a', mu=0 , tau=atau, shape=NxLvl)
    mu = a0 + a
    # define the likelihood
    yl = pm.Normal('yl', mu[x.values], tau=tau, observed=z)
    # Generate a MCMC chain
    start = pm.find_MAP()
    steps = pm.Metropolis()
    trace = pm.sample(20000, steps, start, progressbar=False)


# EXAMINE THE RESULTS
burnin = 2000
thin = 50

# Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

# Check for mixing and autocorrelation
pm.autocorrplot(trace[burnin::thin], vars=model.unobserved_RVs[:-1])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
pm.traceplot(trace)

a0_sample = trace['a0'][burnin::thin]
a_sample = trace['a'][burnin::thin]
# Convert baseline to the original scale
m_sample = a0_sample.repeat(NxLvl).reshape(len(a0_sample), NxLvl) + a_sample
b0_sample = m_sample.mean(axis=1)
b0_sample = b0_sample * y.std() + y.mean()
# Convert baseline to the original scale
n_sample = b0_sample.repeat(NxLvl).reshape(len(b0_sample), NxLvl)
b_sample = (m_sample - n_sample)
b_sample = b_sample * y.std()



plt.figure(figsize=(20, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plot_post(b_sample[:,i], xlab=r'$\beta1_{%s}$' % i,
              show_mode=False, framealpha=0.5,
              bins=50, title='x:%s' % i)
plt.tight_layout()
plt.savefig('Figure_18.2a.png')


nContrasts = len(contrast_dict)
if nContrasts > 0:
    plt.figure(figsize=(20, 8))
    count = 0
    for key, value in contrast_dict.items():
        contrast = np.dot(b_sample, value)
        plt.subplot(2, 5, count)
        plot_post(contrast, title='Contrast %s' % key, comp_val=0.0, 
                  show_mode=False, framealpha=0.5, 
                  bins=50)
        count += 1
    plt.tight_layout()
    plt.savefig('Figure_18.2b.png')

plt.show()
