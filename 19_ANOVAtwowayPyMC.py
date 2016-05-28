"""
Two way BANOVA
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from plot_post import plot_post
import seaborn as sns  # optional
from theano import tensor as tt


# THE DATA.
# Specify data source:
data_source = ["QianS2007" , "Salary" , "Random" , "Ex19.3"][1]

# Load the data:
if data_source == "QianS2007":
    data_record = pd.read_csv("QianS2007SeaweedData.txt")
    # Logistic transform the COVER value:
    # Used by Appendix 3 of QianS2007 to replicate Ramsey and Schafer (2002).
    data_record['COVER'] = -np.log((100/data_record['COVER']) -1)

    y = data_record['COVER'].values
    x1 = pd.Categorical(data_record['TREAT']).codes
    x1names = data_record['TREAT'].values
    x2 = pd.Categorical(data_record['BLOCK']).codes
    x2names = data_record['BLOCK'].values
    Ntotal = len(y)
    Nx1Lvl = len(set(x1))
    Nx2Lvl = len(set(x2))
    x1contrastDict = {'f_Effect':[1/2, -1/2, 0, 1/2, -1/2, 0],
                     'F_Effect':[0, 1/2, -1/2, 0, 1/2, -1/2],
                     'L_Effect':[1/3, 1/3, 1/3, -1/3, -1/3, -1/3 ]}
    x2contrastDict = None # np.zeros(Nx2Lvl)
    x1x2contrastDict = None # np.zeros(Nx1Lvl*Nx2Lvl, Nx1Lvl)

if data_source == "Salary":
    data_record = pd.read_csv("Salary.csv")
    y = data_record['Salary']
    x1 = pd.Categorical(data_record['Org']).codes
    x1names = data_record['Org'].unique()
    x1names.sort()
    x2 = pd.Categorical(data_record['Post']).codes
    x2names = data_record['Post'].unique()
    x2names.sort()
    Ntotal = len(y)
    Nx1Lvl = len(set(x1))
    Nx2Lvl = len(set(x2))

    x1contrastDict = {'BFINvCEDP':[1, -1, 0, 0],
                      'CEDPvTHTR':[0, 1, 0, -1]}
    x2contrastDict = {'FT1vFT2':[1, -1, 0], 
                      'FT2vFT3':[0,1,-1]}
    x1x2contrastDict = {'CHEMvTHTRxFT1vFT3':np.outer([0, 0, 1, -1], [1,0,-1]),
           'BFINvOTHxFT1vOTH':np.outer([1, -1/3, -1/3, -1/3], [1, -1/2, -1/2])}

if data_source == "Random":
    np.random.seed(47405)
    ysdtrue = 3
    a0true = 100
    a1true = np.array([2, 0, -2])  # sum to zero
    a2true = np.array([3, 1, -1, -3])  # sum to zero
    a1a2true = np.array([[1,-1,0, 0], [-1,1,0,0], [0,0,0,0]])
    
    npercell = 8
    index = np.arange(len(a1true)*len(a2true)*npercell)
    data_record = pd.DataFrame(index=index, columns=["y","x1","x2"])

    rowidx = 0
    for x1idx in range(0, len(a1true)):
        for x2idx in range(0, len(a2true)):
            for subjidx in range(0, npercell):
                data_record['x1'][rowidx] = x1idx
                data_record['x2'][rowidx] = x2idx
                data_record['y'][rowidx] = float(a0true + a1true[x1idx] + a2true[x2idx] 
                + a1a2true[x1idx, x2idx] + norm.rvs(loc=0, scale=ysdtrue, size=1)[0])
                rowidx += 1

    y = data_record['y']
    x1 = pd.Categorical(data_record['x1']).codes
    x1names = data_record['x1'].unique()
    x2 = pd.Categorical(data_record['x2']).codes
    x2names = data_record['x2'].unique()
    Ntotal = len(y)
    Nx1Lvl = len(set(x1))
    Nx2Lvl = len(set(x2))
    x1contrast_dict = {'X1_1v3': [1, 0, -1]} #
    x2contrast_dict =  {'X2_12v34':[1/2, 1/2, -1/2, -1/2]} #
    x1x2contrast_dict = {'IC_11v22': np.outer([1, -1, 0], [1, -1, 0, 0]),
    'IC_23v34': np.outer([0, 1, -1], [0, 0, 1, -1])}
    
if data_source == 'Ex19.3':
    y =  [101,102,103,105,104, 104,105,107,106,108, 105,107,106,108,109, 109,108,110,111,112]
    x1 = [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1, 1,1,1,1,1]
    x2 = [0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 1,1,1,1,1]
    S = [0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4]
    x1names = ['x1.1' ,'x1.2']
    x2names = ['x2.1', "x2.2"]
    Snames = ['S1', 'S2', 'S3', 'S4', 'S5']
    Ntotal = len(y)
    Nx1Lvl = len(set(x1))
    Nx2Lvl = len(set(x2))
    NSLvl = len(set(S))
    x1contrast_dict = {'X1.2vX1.1':[-1 , 1]}
    x2contrast_dict = {'X2.2vX2.1':[-1 , 1]}
    x1x2contrast_dict = None #np.arange(0, Nx1Lvl*Nx2Lvl).reshape(Nx1Lvl, -1).T

z = (y - np.mean(y))/np.std(y)
    
z = (y - np.mean(y))/np.std(y)

# THE MODEL.

with pm.Model() as model:
    # define the hyperpriors
    a1_SD_unabs = pm.StudentT('a1_SD_unabs', mu=0, lam=0.001, nu=1)
    a1_SD = abs(a1_SD_unabs) + 0.1
    a1tau = 1 / a1_SD**2

    a2_SD_unabs = pm.StudentT('a2_SD_unabs', mu=0, lam=0.001, nu=1)
    a2_SD = abs(a2_SD_unabs) + 0.1
    a2tau = 1 / a2_SD**2
    
    a1a2_SD_unabs = pm.StudentT('a1a2_SD_unabs', mu=0, lam=0.001, nu=1)
    a1a2_SD = abs(a1a2_SD_unabs) + 0.1
    a1a2tau = 1 / a1a2_SD**2


    # define the priors
    sigma = pm.Uniform('sigma', 0, 10) # y values are assumed to be standardized
    tau = 1 / sigma**2
    
    a0 = pm.Normal('a0', mu=0, tau=0.001) # y values are assumed to be standardized
   
    a1 = pm.Normal('a1', mu=0 , tau=a1tau, shape=Nx1Lvl)
    a2 = pm.Normal('a2', mu=0 , tau=a2tau, shape=Nx2Lvl)
    a1a2 = pm.Normal('a1a2', mu=0 , tau=a1a2tau, shape=[Nx1Lvl, Nx2Lvl])

    b1 = pm.Deterministic('b1', a1 - tt.mean(a1))
    b2 = pm.Deterministic('b2', a2 - tt.mean(a2))
    b1b2 = pm.Deterministic('b1b2', a1a2 - tt.mean(a1a2))
    
    mu = a0 + b1[x1] + b2[x2] + b1b2[x1, x2]
 
    # define the likelihood
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=z)

    # Generate a MCMC chain
    start = pm.find_MAP()
    steps = pm.Metropolis()
    trace = pm.sample(20000, steps, start=start, progressbar=False)

# EXAMINE THE RESULTS
burnin = 2000

# Print summary for each trace
#pm.summary(trace[burnin:])
#pm.summary(trace)

# Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin:], vars=model.unobserved_RVs[:-1])

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace[burnin:])
#pm.traceplot(trace)


# Extract values of 'a'
a0_sample = trace['a0'][burnin:]
b1_sample = trace['b1'][burnin:]
b2_sample = trace['b2'][burnin:]
b1b2_sample = trace['b1b2'][burnin:]

b0_sample = a0_sample * np.std(y) + np.mean(y)
b1_sample = b1_sample * np.std(y)
b2_sample = b2_sample * np.std(y)
b1b2_sample = b1b2_sample * np.std(y)



plt.figure(figsize=(25,20))
plt.subplot(451)
plot_post(b0_sample, xlab=r'$\beta0$',
              show_mode=False, framealpha=0.5,
              bins=50, title='Baseline')
plt.xlim(b0_sample.min(), b0_sample.max());

count = 2
for i in range(len(b1_sample[0])):
    plt.subplot(4, 5, count)
    plot_post(b1_sample[:,i], xlab=r'$\beta1_{%s}$' % i,
              show_mode=False, framealpha=0.5,
              bins=50, title='x1: %s' % x1names[i])
    count += 1

for i in range(len(b2_sample[0])):
    plt.subplot(4, 5, count)
    plot_post(b2_sample[:,i], xlab=r'$\beta2_{%s}$' % i,
              show_mode=False, framealpha=0.5,
              bins=50, title='x1: %s' % x2names[i])    
    count += 1
    
    for j in range(len(b1_sample[0])):
        plt.subplot(4, 5, count)
        plot_post(b1b2_sample[:,j,i], xlab=r'$\beta12_{%s%s}$' % (i, j),
              show_mode=False, framealpha=0.5,
              bins=50, title='x1: %s, x2: %s,' % (x1names[j], x2names[i]))
        count += 1


plt.tight_layout()
plt.savefig('Figure_19.4.png')

## Display contrast analyses
plt.figure(figsize=(10, 12))
count = 1
for key, value in x1contrastDict.items():
    contrast = np.dot(b1_sample, value)
    plt.subplot(3, 2, count)
    plot_post(contrast, title='Contrast %s' % key, comp_val=0.0, 
                  show_mode=False, framealpha=0.5, 
                  bins=50)
    count += 1
    
for key, value in x2contrastDict.items():
    contrast = np.dot(b2_sample, value)
    plt.subplot(3, 2, count)
    plot_post(contrast, title='Contrast %s' % key, comp_val=0.0, 
                  show_mode=False, framealpha=0.5, 
                  bins=50)
    count += 1
    
for key, value in x1x2contrastDict.items():
    contrast = np.tensordot(b1b2_sample, value)
    plt.subplot(3, 2, count)
    plot_post(contrast, title='Contrast %s' % key, comp_val=0.0, 
                  show_mode=False, framealpha=0.5, 
                  bins=50)
    count += 1
plt.tight_layout()
plt.savefig('Figure_19.5.png')

plt.show()
