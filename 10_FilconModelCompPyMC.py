"""
Comparing models using Hierarchical modelling.
"""
from __future__ import division
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# THE DATA.
# For each subject, specify the condition s/he was in,
# the number of trials s/he experienced, and the number correct.

cond_of_subj = np.repeat([0,1,2,3], 40)

n_trl_of_subj = np.array([64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64])

n_corr_of_subj = np.array([45,63,58,64,58,63,51,60,59,47,63,61,60,51,59,45,61,
59,60,58,63,56,63,64,64,60,64,62,49,64,64,58,64,52,64,64,64,62,64,61,59,59,
55,62,51,58,55,54,59,57,58,60,54,42,59,57,59,53,53,42,59,57,29,36,51,64,60,
54,54,38,61,60,61,60,62,55,38,43,58,60,44,44,32,56,43,36,38,48,32,40,40,34,
45,42,41,32,48,36,29,37,53,55,50,47,46,44,50,56,58,42,58,54,57,54,51,49,52,
51,49,51,46,46,42,49,46,56,42,53,55,51,55,49,53,55,40,46,56,47,54,54,42,34,
35,41,48,46,39,55,30,49,27,51,41,36,45,41,53,32,43,33])

n_subj = len(cond_of_subj)
n_cond = len(set(cond_of_subj))


# THE MODEL.
with pm.Model() as model:
    # Hyperprior on model index:
    model_index = pm.DiscreteUniform('model_index', lower=0, upper=1)
    # Constants for hyperprior:
    shape_Gamma = 1.0
    rate_Gamma = 0.1
    # Hyperprior on mu and kappa:
    mu = pm.Beta('mu', 1, 1, shape=n_cond)

    kappa0 = pm.Gamma('kappa0', alpha=shape_Gamma, beta=rate_Gamma)
    a_Beta0 = mu[cond_of_subj] * kappa0
    b_Beta0 = (1 - mu[cond_of_subj]) * kappa0

    kappa1 = pm.Gamma('kappa1', alpha=shape_Gamma, beta=rate_Gamma, shape=n_cond)
    a_Beta1 = mu[cond_of_subj] * kappa1[cond_of_subj]
    b_Beta1 = (1 - mu[cond_of_subj]) * kappa1[cond_of_subj]

    #Prior on theta
    theta0 = pm.Beta('theta0', a_Beta0, b_Beta0, shape=n_subj)
    theta1 = pm.Beta('theta1', a_Beta1, b_Beta1, shape=n_subj)
    # if model_index == 0 then sample from theta1 else sample from theta0
    theta = pm.switch(pm.eq(model_index, 0), theta1, theta0)

    # Likelihood:
    y = pm.Binomial('y', p=theta, n=n_trl_of_subj, observed=n_corr_of_subj)

    # Sampling
    step1 = pm.Metropolis([kappa0, kappa1, mu])
    step2 = pm.NUTS([theta0, theta1])
    step3 = pm.ElemwiseCategorical(vars=[model_index],values=[0,1])
    trace = pm.sample(5000, step=[step1, step2, step3], progressbar=False)


# EXAMINE THE RESULTS.
burnin = 500
pm.traceplot(trace)

model_idx_sample = trace['model_index'][burnin:]

pM1 = sum(model_idx_sample == 1) / len(model_idx_sample)
pM2 = 1 - pM1

plt.figure(figsize=(15, 15))
plt.subplot2grid((5,4), (0,0), colspan=4)
plt.plot(model_idx_sample, label='p(M1|D) = %.3f ; p(M2|D) = %.3f' % (pM1, pM2));
plt.xlabel('Steps in Markov Chain')
plt.legend(loc='upper right', framealpha=0.75)

for m in range(0, 2):
    kappa0_sample = trace['kappa0'][burnin:][model_idx_sample == m]
    plt.subplot2grid((5,4), (3+m, 1), colspan=2)
    plt.hist(kappa0_sample, bins=30)
    plt.title(r'Post. $\kappa_0$ for M=%s' % (m+1), fontsize=14)
    plt.xlabel(r'$\kappa_0$')
    plt.xlim(0, 30)
    for i in range(0, 4):
        kappa1_sample = trace['kappa1'][:,i][burnin:][model_idx_sample == m]
        plt.subplot2grid((5,4), (m+1, i))
        plt.hist(kappa1_sample, bins=30)
        plt.title(r'Post. $\kappa_%s$ for M=%s' % (i+1, m+1), fontsize=14)
        plt.xlabel(r'$\kappa_%s$' % (i+1))
        plt.xlim(0, 30)

plt.tight_layout()
plt.savefig('Figure_10.3-4.png')
plt.show()
