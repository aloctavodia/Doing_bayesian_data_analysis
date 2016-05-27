"""
Inferring a binomial proportion using PyMC.
"""
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

# Generate the data
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # 11 heads and 3 tails


with pm.Model() as model:
    # define the prior
    theta = pm.Beta('theta', 1, 1)  # prior
    # define the likelihood
    y = pm.Bernoulli('y', p=theta, observed=y)

    # Generate a MCMC chain
    trace = pm.sample(5000, pm.Metropolis(),
                      progressbar=False)  # Use Metropolis sampling
#    start = pm.find_MAP()  # Find starting value by optimization
#    step = pm.NUTS()  # Instantiate NUTS sampler
#    trace = pm.sample(5000, step, start=start, progressbar=False)

# create an array with the posterior sample
theta_sample = trace['theta']

fig, ax = plt.subplots(1, 2)
ax[0].plot(theta_sample[:500], np.arange(500), marker='o', color='skyblue')
ax[0].set_xlim(0, 1)
ax[0].set_xlabel(r'$\theta$')
ax[0].set_ylabel('Position in Chain')

pm.plot_posterior(theta_sample, ax=ax[1], color='skyblue');
ax[1].set_xlabel(r'$\theta$');

# Posterior prediction:
# For each step in the chain, use posterior theta to flip a coin:
y_pred = np.zeros(len(theta_sample))
for i, p_head in enumerate(theta_sample):
    y_pred[i] = np.random.choice([0, 1], p=[1 - p_head, p_head])

# Jitter the 0,1 y values for plotting purposes:
y_pred_jittered = y_pred + np.random.uniform(-.05, .05, size=len(theta_sample))

# Now plot the jittered values:
plt.figure()
plt.plot(theta_sample[:500], y_pred_jittered[:500], 'ro')
plt.xlim(-.1, 1.1)
plt.ylim(-.1, 1.1)
plt.xlabel(r'$\theta$')
plt.ylabel('y (jittered)')

mean_y = np.mean(y_pred)
mean_theta = np.mean(theta_sample)

plt.plot(mean_y, mean_theta, 'k+', markersize=15)
plt.annotate('mean(y) = %.2f\nmean($\\theta$) = %.2f' %
    (mean_y, mean_theta), xy=(mean_y, mean_theta))
plt.plot([0, 1], [0, 1], linestyle='--')

plt.savefig('BernBetaPyMCPost.png')
plt.show()
