"""
Inferring a binomial proportion via exact mathematical analysis.
"""
import sys
import numpy as np
from scipy.stats import beta
from scipy.special import beta as beta_func
import matplotlib.pyplot as plt
from HDIofICDF import *


def bern_beta(prior_shape, data_vec, cred_mass=0.95):
    """Bayesian updating for Bernoulli likelihood and beta prior.
     Input arguments:
       prior_shape
         vector of parameter values for the prior beta distribution.
       data_vec
         vector of 1's and 0's.
       cred_mass
         the probability mass of the HDI.
     Output:
       post_shape
         vector of parameter values for the posterior beta distribution.
     Graphics:
       Creates a three-panel graph of prior, likelihood, and posterior
       with highest posterior density interval.
     Example of use:
     post_shape = bern_beta(prior_shape=[1,1] , data_vec=[1,0,0,1,1])"""

    # Check for errors in input arguments:
    if len(prior_shape) != 2:
        sys.exit('prior_shape must have two components.')
    if any([i < 0 for i in prior_shape]):
        sys.exit('prior_shape components must be positive.')
    if any([i != 0 and i != 1 for i in data_vec]):
        sys.exit('data_vec must be a vector of 1s and 0s.')
    if cred_mass <= 0 or cred_mass >= 1.0:
        sys.exit('cred_mass must be between 0 and 1.')

    # Rename the prior shape parameters, for convenience:
    a = prior_shape[0]
    b = prior_shape[1]
    # Create summary values of the data:
    z = sum(data_vec[data_vec == 1])  # number of 1's in data_vec
    N = len(data_vec)   # number of flips in data_vec
    # Compute the posterior shape parameters:
    post_shape = [a+z, b+N-z]
    # Compute the evidence, p(D):
    p_data = beta_func(z+a, N-z+b)/beta_func(a, b)
    # Construct grid of theta values, used for graphing.
    bin_width = 0.005  # Arbitrary small value for comb on theta.
    theta = np.arange(bin_width/2, 1-(bin_width/2)+bin_width, bin_width)
    # Compute the prior at each value of theta.
    p_theta = beta.pdf(theta, a, b)
    # Compute the likelihood of the data at each value of theta.
    p_data_given_theta = theta**z * (1-theta)**(N-z)
    # Compute the posterior at each value of theta.
    post_a = a + z
    post_b = b+N-z
    p_theta_given_data = beta.pdf(theta, a+z, b+N-z)
    # Determine the limits of the highest density interval
    intervals = HDIofICDF(beta, cred_mass, a=post_shape[0], b=post_shape[1])

    # Plot the results.
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.7)

    # Plot the prior.
    locx = 0.05
    plt.subplot(3, 1, 1)
    plt.plot(theta, p_theta)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_theta)*1.2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(\theta)$')
    plt.title('Prior')
    plt.text(locx, np.max(p_theta)/2, r'beta($\theta$;%s,%s)' % (a, b))
    # Plot the likelihood:
    plt.subplot(3, 1, 2)
    plt.plot(theta, p_data_given_theta)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_data_given_theta)*1.2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(D|\theta)$')
    plt.title('Likelihood')
    plt.text(locx, np.max(p_data_given_theta)/2, 'Data: z=%s, N=%s' % (z, N))
    # Plot the posterior:
    plt.subplot(3, 1, 3)
    plt.plot(theta, p_theta_given_data)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_theta_given_data)*1.2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(\theta|D)$')
    plt.title('Posterior')
    locy = np.linspace(0, np.max(p_theta_given_data), 5)
    plt.text(locx, locy[1], r'beta($\theta$;%s,%s)' % (post_a, post_b))
    plt.text(locx, locy[2], 'P(D) = %g' % p_data)
    # Plot the HDI
    plt.text(locx, locy[3],
             'Intervals = %.3f - %.3f' % (intervals[0], intervals[1]))
    plt.fill_between(theta, 0, p_theta_given_data,
                    where=np.logical_and(theta > intervals[0],
                    theta < intervals[1]),
                        color='blue', alpha=0.3)
    return intervals

data_vec = np.repeat([1, 0], [11, 3])  # 11 heads, 3 tail
intervals = bern_beta(prior_shape=[100, 100], data_vec=data_vec)
plt.savefig('Figure_5.2.png')
plt.show()

