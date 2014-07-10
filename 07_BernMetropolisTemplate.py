"""
Use this program as a template for experimenting with the Metropolis algorithm
applied to a single parameter called theta, defined on the interval [0,1].
"""
from __future__ import division
import numpy as np
from scipy.stats import beta
from plot_post import *



# Specify the data, to be used in the likelihood function.
# This is a vector with one component per flip,
# in which 1 means a "head" and 0 means a "tail".
my_data = np.repeat([1, 0], [11, 3])  # 11 heads, 2 tail

# Define the Bernoulli likelihood function, p(D|theta).
# The argument theta could be a vector, not just a scalar.
def likelihood(theta, data):
    theta = np.array(theta) # ensure you have an array
    z = sum(data[data == 1])  # number of 1's in Data
    N = len(data)  # number of flips in Data
# Compute the likelihood of the Data for each value of Theta.
    if np.size(theta) == 1:  # if theta is an scalar
        p_data_given_theta = 0
        if theta < 1 and theta > 0:
            p_data_given_theta = theta**z * (1-theta)**(N-z)
    else: # if theta is an array
        p_data_given_theta = theta**z * (1-theta)**(N-z)
        # The theta values passed into this function are generated at random,
        # and therefore might be inadvertently greater than 1 or less than 0.
        # The likelihood for theta > 1 or for theta < 0 is zero:
        p_data_given_theta[(theta > 1) | (theta < 0)] = 0
    return p_data_given_theta


# Define the prior density function. For purposes of computing p(D),
# at the end of this program, we want this prior to be a proper density.
# The argument theta could be a vector, not just a scalar.
def prior(theta):
    theta = np.array(theta) # ensure you have an array
# For kicks, here's a bimodal prior. To try it, uncomment the next 2 lines.
    #from scipy.stats import beta
    #prior = dbeta(np.minium(2*theta, 2*(1-theta)), 2, 2)
    if np.size(theta) == 1:  # if theta is an scalar
        prior = 0
        if theta < 1 and theta > 0:
            prior = 1
    else: # if theta is an array
        prior = np.ones(len(theta))  # uniform density over [0,1]
        # The theta values passed into this function are generated at random,
        # and therefore might be inadvertently greater than 1 or less than 0.
        # The likelihood for theta > 1 or for theta < 0 is zero:
        prior[(theta > 1) | (theta < 0)] = 0
    return prior



# Define the relative probability of the target distribution, 
# as a function of vector theta. For our application, this
# target distribution is the unnormalized posterior distribution.
def target_rel_prob(theta, data):
    target_rel_prob = likelihood(theta , data) * prior(theta)
    return target_rel_prob

# Specify the length of the trajectory, i.e., the number of jumps to try:
traj_length = 5000 # arbitrary large number
# Initialize the vector that will store the results:
trajectory = np.zeros(traj_length)
# Specify where to start the trajectory:
trajectory[0] = 0.50 # arbitrary value
# Specify the burn-in period:
burn_in = np.ceil(0.1 * traj_length) # arbitrary number, less than traj_length
# Initialize accepted, rejected counters, just to monitor performance:
n_accepted = 0
n_rejected = 0
# Specify seed to reproduce same random walk:
np.random.seed(4745)

# Now generate the random walk. The 't' index is time or trial in the walk.
for t in range(traj_length-1):
    current_position = trajectory[t]
    # Use the proposal distribution to generate a proposed jump.
    # The shape and variance of the proposal distribution can be changed
    # to whatever you think is appropriate for the target distribution.
    proposed_jump = np.random.normal(loc=0 , scale=0.1, size=1)
    
#    # Compute the probability of accepting the proposed jump.
    prob_accept = np.minimum(1, 
                            target_rel_prob(current_position + proposed_jump, my_data)
                            / target_rel_prob(current_position, my_data))
#    # Generate a random uniform value from the interval [0,1] to
#    # decide whether or not to accept the proposed jump.
    if np.random.rand() < prob_accept:
        # accept the proposed jump
        trajectory[t+1] = current_position + proposed_jump
        # increment the accepted counter, just to monitor performance
        if t > burn_in:
            n_accepted += 1
    else:
        # reject the proposed jump, stay at current position
        trajectory[t+1] = current_position
        # increment the rejected counter, just to monitor performance
        if t > burn_in:
            n_rejected += 1


# Extract the post-burn_in portion of the trajectory.
accepted_traj = trajectory[burn_in:]
# End of Metropolis algorithm.



# Display rejected/accepted ratio in the plot.
mean_traj = np.mean(accepted_traj)
std_traj = np.std(accepted_traj)
plt.plot(0, label=r'$N_{pro}=%s$ $\frac{N_{acc}}{N_{pro}} = %.3f$' % (len(accepted_traj), (n_accepted/len(accepted_traj))), alpha=0)

# Evidence for model, p(D).

# Compute a,b parameters for beta distribution that has the same mean
# and stdev as the sample from the posterior. This is a useful choice
# when the likelihood function is Bernoulli.
a =   mean_traj   * ((mean_traj*(1 - mean_traj)/std_traj**2) - 1)
b = (1 - mean_traj) * ((mean_traj*(1 - mean_traj)/std_traj**2) - 1)

# For every theta value in the posterior sample, compute 
# dbeta(theta,a,b) / likelihood(theta)*prior(theta)
# This computation assumes that likelihood and prior are proper densities,
# i.e., not just relative probabilities. This computation also assumes that
# the likelihood and prior functions were defined to accept a vector argument,
# not just a single-component scalar argument.
wtd_evid = beta.pdf(accepted_traj, a, b) / (likelihood(accepted_traj, my_data) * prior(accepted_traj))
p_data = 1 / np.mean(wtd_evid)


# Display p(D) in the graph
plt.plot(0, label='p(D) = %.3e' % p_data, alpha=0)

# Display the posterior.
ROPE = np.array([0.76, 0.8])
mcmc_info = plot_post(accepted_traj, xlab='theta', show_mode=False, comp_val=0.9, ROPE=ROPE)


# Uncomment next line if you want to save the graph.
plt.savefig('BernMetropolisTemplate.png')
plt.show()
