"""
Inferring a binomial proportion via grid aproximation.
"""
import matplotlib.pyplot as plt
import numpy as np
from hpd import hpd


def bern_grid(theta, p_theta, data, credib=.95):
    """
    Bayesian updating for Bernoulli likelihood and prior specified on a grid.
    Input arguments:
     theta is a vector of theta values, all between 0 and 1.
     p_theta is a vector of corresponding probability _masses_.
     data is a vector of 1's and 0's, where 1 corresponds to a and 0 to b.
     credib is the probability mass of the credible interval, default is 0.95.
    Output:
     p_theta_given_data is a vector of posterior probability masses over theta.
     Also creates a three-panel graph of prior, likelihood, and posterior
     probability masses with credible interval.
    Example of use:
     Create vector of theta values.
     bin_width = 1/1000 
     theta_grid = np.arange(0, 1+bin_width, bin_width)
     Specify probability mass at each theta value.
     > rel_prob = np.minimum(theta_grid, 1-theta_grid) relative prob at each theta
     > prior = rel_prob / sum(rel_prob) probability mass at each theta
     Specify the data vector.
     data_vec = np.repeat([1, 0], [11, 3])  # 3 heads, 1 tail
     Call the function.
     > posterior = bern_grid( theta=theta_grid , p_theta=prior , data=data_vec )
    """

# Create summary values of data
    z = sum(data[data == 1])  # number of 1's in data
    N = len(data)  # number of flips in data
# Compute the likelihood of the data for each value of theta.
    p_data_given_theta = theta**z * (1 - theta)**(N - z)
# Compute the evidence and the posterior.
    p_data = sum(p_data_given_theta * p_theta)
    p_theta_given_data = p_data_given_theta * p_theta / p_data
    # Determine the limits of the highest density interval
    x = np.random.choice(theta, size=5000, replace=True, p=p_theta_given_data)
    intervals = hpd(x, alpha=1-credib)

# Plot the results.
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.7)

#    # Plot the prior.
    locx = 0.05
    mean_theta = sum(theta * p_theta)  # mean of prior, for plotting
    plt.subplot(3, 1, 1)
    plt.plot(theta, p_theta)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_theta)*1.2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(\theta)$')
    plt.title('Prior')
    plt.text(locx, np.max(p_theta)/2, r'mean($\theta$;%5.2f)' % mean_theta)
    # Plot the likelihood:
    plt.subplot(3, 1, 2)
    plt.plot(theta, p_data_given_theta)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_data_given_theta)*1.2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(D|\theta)$')
    plt.title('Likelihood')
    plt.text(locx, np.max(p_data_given_theta)/2, 'data: z=%s, N=%s' % (z, N))
    # Plot the posterior:
    mean_theta_given_data = sum(theta * p_theta_given_data)
    plt.subplot(3, 1, 3)
    plt.plot(theta, p_theta_given_data)
    plt.xlim(0, 1)
    plt.ylim(0, np.max(p_theta_given_data)*1.2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(\theta|D)$')
    plt.title('Posterior')
    loc = np.linspace(0, np.max(p_theta_given_data), 5)
    plt.text(locx, loc[1], r'mean($\theta$;%5.2f)' % mean_theta_given_data)
    plt.text(locx, loc[2], 'P(D) = %g' % p_data)
    # Plot the HDI
    plt.text(locx, loc[3],
             'Intervals =%s' % ', '.join('%.3f' % x for x in intervals))
    for i in range(0, len(intervals), 2):
        plt.fill_between(theta, 0, p_theta_given_data,
                         where=np.logical_and(theta > intervals[i],
                                              theta < intervals[i+1]),
                         color='blue', alpha=0.3)
    plt.savefig('Figure_6.1.png')
    plt.show()
    return p_theta_given_data


###Create vector of theta values.
bin_width = 1/1000.
theta_grid = np.arange(0, 1+bin_width, bin_width)
##Specify probability mass at each theta value.
rel_prob = np.array([0.1] * len(theta_grid))  # uniform prior
rel_prob = np.array([0.1] * len(theta_grid))  # uniform prior
prior = rel_prob / sum(rel_prob)  # probability mass at each theta


#### figure 6.2 ###
#np.random.seed(123)
#a = [0.1] * 50
#b = np.linspace(0.1, 1, 50)
#c = np.linspace(1, 0.1, 50)
#d = [0.1] * 50
#p_theta = np.concatenate((a, b, c, d))
#prior = np.where(p_theta != 0 , p_theta / sum(p_theta), 0.)
#width = 1. / len(p_theta)
#theta_grid = np.arange(width/2 , (1-width/2)+width, width)

### figure 6.3 ###
#np.random.seed(123)
#a = np.repeat([0], [50])
#b = np.linspace(0, 1, 50)
#c = (np.linspace(1, 0, 20))**2
#d = np.random.uniform(size=3)
#e = np.repeat([1], [20])
#p_theta = np.concatenate((a, b, c, d, e))
#prior = np.where(p_theta != 0 , p_theta / sum(p_theta), 0.)
#width = 1. / len(p_theta)
#theta_grid = np.arange(width/2 , (1-width/2)+width, width)

###Specify the data vector.
data_vec = np.repeat([1, 0], [11, 3])  # 3 heads, 1 tail
###Call the function.
posterior = bern_grid(theta=theta_grid, p_theta=prior, data=data_vec)
