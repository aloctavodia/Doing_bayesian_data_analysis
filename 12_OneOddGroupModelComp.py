"""
Testing a point ('Null') Hypothesis (not using pseudopriors)
"""
from __future__ import division
import numpy as np
import pymc3 as pm
from scipy.stats import binom
import matplotlib.pyplot as plt
from plot_post import plot_post

# THE DATA.
# For each subject, specify the condition s/he was in,
# the number of trials s/he experienced, and the number correct.
# (Randomly generated fictitious data.)
npg = 20  # number of subjects per group
ntrl = 20 # number of trials per subject
cond_of_subj = np.repeat([0, 1, 2, 3], npg)
n_trl_of_subj = np.repeat([ntrl], 4*npg)
np.random.seed(47401)

n_corr_of_subj = np.concatenate((binom.rvs(n=ntrl, p=.61, size=npg), 
                binom.rvs(n=ntrl, p=.50, size=npg),
                binom.rvs(n=ntrl, p=.49, size=npg),
                binom.rvs(n=ntrl, p=.51, size=npg)))

n_subj = len(cond_of_subj)
n_cond = len(set(cond_of_subj))


# THE MODEL
with pm.Model() as model:
    # Hyperprior on model index:
    model_index = pm.DiscreteUniform('model_index', lower=0, upper=1)
    # Constants for hyperprior:
    shape_Gamma = 1.0
    rate_Gamma = 0.1
    # Hyperprior on mu and kappa:
    kappa = pm.Gamma('kappa', shape_Gamma, rate_Gamma, shape=n_cond)

    mu0 = pm.Beta('mu0', 1, 1)
    a_Beta0 = mu0 * kappa[cond_of_subj]
    b_Beta0 = (1 - mu0) * kappa[cond_of_subj]

    mu1 = pm.Beta('mu1', 1, 1, shape=n_cond)
    a_Beta1 = mu1[cond_of_subj] * kappa[cond_of_subj]
    b_Beta1 = (1 - mu1[cond_of_subj]) * kappa[cond_of_subj]

    #Prior on theta
    theta0 = pm.Beta('theta0', a_Beta0, b_Beta0, shape=n_subj)
    theta1 = pm.Beta('theta1', a_Beta1, b_Beta1, shape=n_subj)
    # if model_index == 0 then sample from theta1 else sample from theta0
    theta = pm.switch(pm.eq(model_index, 0), theta1, theta0)

    # Likelihood:
    y = pm.Binomial('y', p=theta, n=n_trl_of_subj, observed=n_corr_of_subj)

    # Sampling
    start = pm.find_MAP()
    step1 = pm.Metropolis(model.vars[1:])
    step2 = pm.ElemwiseCategoricalStep(var=model_index,values=[0,1])
    trace = pm.sample(20000, [step1, step2], start=start, progressbar=False)


# EXAMINE THE RESULTS.
burnin = 10000
thin = 10

## Print summary for each trace
#pm.summary(trace[burnin::thin])
#pm.summary(trace)

## Check for mixing and autocorrelation
#pm.autocorrplot(trace, vars =[mu, kappa])

## Plot KDE and sampled values for each parameter.
#pm.traceplot(trace[burnin::thin])
#pm.traceplot(trace)

model_idx_sample = trace['model_index'][burnin::thin]
pM1 = sum(model_idx_sample == 1) / len(model_idx_sample)
pM2 = 1 - pM1

plt.figure(figsize=(15, 15))
plt.subplot2grid((3,3), (0,0), colspan=3)
plt.plot(model_idx_sample, label='p(DiffMu|D) = %.3f ; p(SameMu|D) = %.3f' % (pM1, pM2));
plt.xlabel('Step in Markov Chain')
plt.legend(loc='upper right', framealpha=0.75)

count = 0
position = [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
for i in range(0, 4):
    mui_sample = trace['mu1'][:,i][burnin::thin][model_idx_sample == 0]
    for j in range(i+1, 4):
        muj_sample = trace['mu1'][:,j][burnin::thin][model_idx_sample == 0]
        plt.subplot2grid((3,3), position[count])
        plot_post(mui_sample-muj_sample, xlab=r'$\mu_%s - \mu_%s$' % (i+1, j+1), show_mode=False, comp_val=0, framealpha=0.5)
        plt.xlim(-0.3, 0.3)
        count += 1


plt.tight_layout()
plt.savefig('Figure_12.5.png')
plt.show()

