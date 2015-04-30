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
from hpd import * 
import seaborn as sns
from theano import tensor as T


# THE DATA.
# Specify data source:
dataSource = ["QianS2007" , "Salary" , "Random" , "Ex19.3"][1]


# Load the data:
if dataSource == "QianS2007":
    datarecord = pd.read_csv("QianS2007SeaweedData.txt")
    # Logistic transform the COVER value:
    # Used by Appendix 3 of QianS2007 to replicate Ramsey and Schafer (2002).
    datarecord['COVER'] = -np.log((100/datarecord['COVER']) -1)

    y = datarecord['COVER'].values
    x1 = pd.Categorical(datarecord['TREAT']).codes
    x1names = datarecord['TREAT'].values
    x2 = pd.Categorical(datarecord['BLOCK']).codes
    x2names = datarecord['BLOCK'].values
    Ntotal = len(y)
    Nx1Lvl = len(set(x1))
    Nx2Lvl = len(set(x2))
    x1contrastDict = {'f_Effect':[1/2, -1/2, 0, 1/2, -1/2, 0],
                     'F_Effect':[0, 1/2, -1/2, 0, 1/2, -1/2],
                     'L_Effect':[1/3, 1/3, 1/3, -1/3, -1/3, -1/3 ]}
    x2contrastDict = None # np.zeros(Nx2Lvl)
    x1x2contrastDict = None # np.zeros(Nx1Lvl*Nx2Lvl, Nx1Lvl)

if dataSource == "Salary":
    datarecord = pd.read_csv("Salary.csv")
    y = datarecord['Salary']
    x1 = pd.Categorical(datarecord['Org']).codes
    x1names = datarecord['Org'].unique()
    x1names.sort()
    x2 = pd.Categorical(datarecord['Post']).codes
    x2names = datarecord['Post'].unique()
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
    

        
#if ( data_source == "Random" ) {
#  fnroot = paste( fnroot , data_source , sep="" )
#  set.seed(47405)
#  ysdtrue = 3.0
#  a0true = 100
#  a1true = c( 2 , 0 , -2 ) # sum to zero
#  a2true = c( 3 , 1 , -1 , -3 ) # sum to zero
#  a1a2true = matrix( c( 1,-1,0, -1,1,0, 0,0,0, 0,0,0 ),# row and col sum to zero
#                     nrow=length(a1true) , ncol=length(a2true) , byrow=F )
#  npercell = 8
#  datarecord = matrix( 0, ncol=3 , nrow=length(a1true)*length(a2true)*npercell )
#  colnames(datarecord) = c("y","x1","x2")
#  rowidx = 0
#  for ( x1idx in 1:length(a1true) ) {
#    for ( x2idx in 1:length(a2true) ) {
#      for ( subjidx in 1:npercell ) {
#        rowidx = rowidx + 1
#        datarecord[rowidx,"x1"] = x1idx
#        datarecord[rowidx,"x2"] = x2idx
#        datarecord[rowidx,"y"] = ( a0true + a1true[x1idx] + a2true[x2idx]
#                                 + a1a2true[x1idx,x2idx] + rnorm(1,0,ysdtrue) )
#      }
#    }
#  }
#  datarecord = data.frame( y=datarecord[,"y"] ,
#                           x1=as.factor(datarecord[,"x1"]) ,
#                           x2=as.factor(datarecord[,"x2"]) )
#  y = as.numeric(datarecord$y)
#  x1 = as.numeric(datarecord$x1)
#  x1names = levels(datarecord$x1)
#  x2 = as.numeric(datarecord$x2)
#  x2names = levels(datarecord$x2)
#  Ntotal = length(y)
#  Nx1Lvl = length(unique(x1))
#  Nx2Lvl = length(unique(x2))
#  x1contrast_dict = list( X1_1v3 = c( 1 , 0 , -1 ) ) #
#  x2contrast_dict =  list( X2_12v34 = c( 1/2 , 1/2 , -1/2 , -1/2 ) ) #
#  x1x2contrast_dict = list(
#    IC_11v22 = outer( c(1,-1,0) , c(1,-1,0,0) ) ,
#    IC_23v34 = outer( c(0,1,-1) , c(0,0,1,-1) )
#  )
#}

## Load the data:
#if ( data_source == "Ex19.3" ) {
#  fnroot = paste( fnroot , data_source , sep="" )
#  y = c( 101,102,103,105,104, 104,105,107,106,108, 105,107,106,108,109, 109,108,110,111,112 )
#  x1 = c( 1,1,1,1,1, 1,1,1,1,1, 2,2,2,2,2, 2,2,2,2,2 )
#  x2 = c( 1,1,1,1,1, 2,2,2,2,2, 1,1,1,1,1, 2,2,2,2,2 )
#  # S = c( 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5 )
#  x1names = c("x1.1","x1.2")
#  x2names = c("x2.1","x2.2")
#  # Snames = c("S1","S2","S3","S4","S5")
#  Ntotal = length(y)
#  Nx1Lvl = length(unique(x1))
#  Nx2Lvl = length(unique(x2))
#  # NSLvl = length(unique(S))
#  x1contrast_dict = list( X1.2vX1.1 = c( -1 , 1 ) )
#  x2contrast_dict = list( X2.2vX2.1 = c( -1 , 1 ) )
#  x1x2contrast_dict = NULL # list( matrix( 1:(Nx1Lvl*Nx2Lvl) , nrow=Nx1Lvl ) )
#}

z = (y - np.mean(y))/np.std(y)

# THE MODEL.

with pm.Model() as model:
    # define the hyperpriors
    a1_SD_unabs = pm.T('a1_SD_unabs', mu=0, lam=0.001, nu=1)
    a1_SD = abs(a1_SD_unabs) + 0.1
    a1tau = 1 / a1_SD**2

    a2_SD_unabs = pm.T('a2_SD_unabs', mu=0, lam=0.001, nu=1)
    a2_SD = abs(a2_SD_unabs) + 0.1
    a2tau = 1 / a2_SD**2
    
    a1a2_SD_unabs = pm.T('a1a2_SD_unabs', mu=0, lam=0.001, nu=1)
    a1a2_SD = abs(a1a2_SD_unabs) + 0.1
    a1a2tau = 1 / a1a2_SD**2


    # define the priors
    sigma = pm.Uniform('sigma', 0, 10) # y values are assumed to be standardized
    tau = 1 / sigma**2
    
    a0 = pm.Normal('a0', mu=0, tau=0.001) # y values are assumed to be standardized
   
    a1 = pm.Normal('a1', mu=0 , tau=a1tau, shape=Nx1Lvl)
    a2 = pm.Normal('a2', mu=0 , tau=a2tau, shape=Nx2Lvl)
    a1a2 = pm.Normal('a1a2', mu=0 , tau=a1a2tau, shape=[Nx1Lvl, Nx2Lvl])

    b1 = pm.Deterministic('b1', a1 - T.mean(a1))
    b2 = pm.Deterministic('b2', a2 - T.mean(a2))
    b1b2 = pm.Deterministic('b1b2', a1a2 - T.mean(a1a2))
    
    mu = a0 + b1[x1] + b2[x2] + b1b2[x1, x2]
 
    # define the likelihood
    yl = pm.Normal('yl', mu=mu, tau=tau, observed=z)

    # Generate a MCMC chain
    start = pm.find_MAP()
    steps = pm.Metropolis()
    trace = pm.sample(20000, steps, start=start, progressbar=True)

# EXAMINE THE RESULTS
burnin = 2000
thin = 50

# Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

# Check for mixing and autocorrelation
#pm.autocorrplot(trace[burnin::thin], vars=model.unobserved_RVs[:-1])

## Plot KDE and sampled values for each parameter.
pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)


# Extract values of 'a'
a0_sample = trace['a0'][burnin::thin]
b1_sample = trace['b1'][burnin::thin]
b2_sample = trace['b2'][burnin::thin]
b1b2_sample = trace['b1b2'][burnin::thin]

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
