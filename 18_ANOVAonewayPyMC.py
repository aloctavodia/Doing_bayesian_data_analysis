"""
One way BANOVA
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from plot_post import plot_post
from hpd import * 
import seaborn as sns
from theano import tensor as T


# THE DATA.
# Specify data source:
dataSource = ["McDonaldSK1991" , "SolariLS2008" , "Random"][0]

# Load the data:
if dataSource == "McDonaldSK1991":
    datarecord = pd.read_csv("McDonaldSK1991data.txt", sep='\s+', skiprows=18, skipfooter=25)
    y = datarecord['Size']
    Ntotal = len(y)
    x = (datarecord['Group'] - 1).values
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


if dataSource == "SolariLS2008":
    datarecord = pd.read_csv("SolariLS2008data.txt", sep='\s+', skiprows=21)
    y = datarecord['Acid']
    Ntotal = len(y)
    x = (datarecord['Type'] - 1).values
    xnames = pd.unique(x)
    NxLvl = len(xnames)
    contrast_dict = {'G3vOTHER':[-1/8,-1/8,1,-1/8,-1/8,-1/8,-1/8,-1/8,-1/8]}


if dataSource == "Random":
    np.random.seed(47405)
    ysdtrue = 4.0
    a0true = 100
    atrue = [2, -2]  # sum to zero
    npercell = 8
    x = []
    y = []
    for xidx in range(len(atrue)):
        for subjidx in range(npercell):
            x.append(xidx)
            y.append(a0true + atrue[xidx] + norm.rvs(1, ysdtrue))
    Ntotal = len(y)
    NxLvl = len(set(x))
#  # Construct list of all pairwise comparisons, to compare with NHST TukeyHSD:
    contrast_dict = None
    for g1idx in range(NxLvl):
        for g2idx in range(g1idx+1, NxLvl):
            cmpVec = np.repeat(0, NxLvl)
            cmpVec[g1idx] = -1
            cmpVec[g2idx] = 1
            contrast_dict = (contrast_dict, cmpVec)


z = (y - np.mean(y))/np.std(y)


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
    
    b = pm.Deterministic('b', a - T.mean(a))
    mu = a0 + b[x]
    # define the likelihood
    yl = pm.Normal('yl', mu, tau=tau, observed=z)
    # Generate a MCMC chain
    start = pm.find_MAP()
    steps = pm.Metropolis()
    trace = pm.sample(20000, steps, start, progressbar=False)


# EXAMINE THE RESULTS
burnin = 1000
thin = 10

# Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

# Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars=model.unobserved_RVs[:-1])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
pm.traceplot(trace)

a0_sample = trace['a0'][burnin::thin]
b_sample = trace['b'][burnin::thin]
b0_sample = a0_sample * np.std(y) + np.mean(y)
b_sample = b_sample * np.std(y)


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
        plt.subplot(2, 4, count)
        plot_post(contrast, title='Contrast %s' % key, comp_val=0.0, 
                  show_mode=False, framealpha=0.5, 
                  bins=50)
        count += 1
    plt.tight_layout()
    plt.savefig('Figure_18.2b.png')

plt.show()
