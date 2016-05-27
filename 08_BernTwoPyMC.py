"""
Inferring two binomial proportions using PyMC.
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm


# Generate the data
y1 = np.array([1, 1, 1, 1, 1, 0, 0])  # 5 heads and 2 tails
y2 = np.array([1, 1, 0, 0, 0, 0, 0])  # 2 heads and 5 tails


with pm.Model() as model:
    # define the prior
    theta1 = pm.Beta('theta1', 3, 3)  # prior
    theta2 = pm.Beta('theta2', 3, 3)  # prior
    # define the likelihood
    y1 = pm.Bernoulli('y1', p=theta1, observed=y1)
    y2 = pm.Bernoulli('y2', p=theta2, observed=y2)

    # Generate a MCMC chain
    start = pm.find_MAP()  # Find starting value by optimization
    trace = pm.sample(10000, pm.Metropolis(),
                      progressbar=False)  # Use Metropolis sampling
#    start = pm.find_MAP()  # Find starting value by optimization
#    step = pm.NUTS()  # Instantiate NUTS sampler
#    trace = pm.sample(10000, step, start=start, progressbar=False)

# create an array with the posterior sample
theta1_sample = trace['theta1']
theta2_sample = trace['theta2']

# Plot the trajectory of the last 500 sampled values.
plt.plot(theta1_sample[:-500], theta2_sample[:-500], marker='o',  color='skyblue')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\theta2$')

# Display means in plot.
plt.plot(0, label='M = %.3f, %.3f' % (np.mean(theta1_sample), np.mean(theta2_sample)), alpha=0.0)

plt.legend(loc='upper left')
plt.savefig('Figure_8.6.png')

# Plot a histogram of the posterior differences of theta values.
theta_diff = theta1_sample - theta2_sample
pm.plot_posterior(theta_diff, ref_val=0.0, bins=30, color='skyblue')
plt.xlabel(r'$\theta_1 - \theta_2$')
plt.savefig('Figure_8.8.png')

# For Exercise 8.5:
# Posterior prediction. For each step in the chain, use the posterior thetas 
# to flip the coins.
chain_len = len(theta1_sample)
# Create matrix to hold results of simulated flips:
y_pred = np.zeros((2, chain_len))
for step_idx in range(chain_len):  # step through the chain
    # flip the first coin:
    p_head1 = theta1_sample[step_idx]
    y_pred[0, step_idx] = np.random.choice([0,1], p=[1-p_head1, p_head1])
    # flip the second coin:
    p_head2 = theta2_sample[step_idx]
    y_pred[1, step_idx] = np.random.choice([0,1], p=[1-p_head2, p_head2])


# Now determine the proportion of times that y1==1 and y2==0
pY1eq1andY2eq0 = sum((y_pred[0] ==1) & (y_pred[1] == 0)) / chain_len

print(pY1eq1andY2eq0)
plt.show()

