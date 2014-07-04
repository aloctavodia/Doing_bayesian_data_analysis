"""
Goal: Toss a coin N times and compute the running proportion of heads.
"""
import matplotlib.pyplot as plt
import numpy as np

# Specify the total number of flips, denoted N.
N = 500
# Generate a random sample of N flips for a fair coin (heads=1, tails=0);
np.random.seed(47405)
flip_sequence = np.random.choice(a=(0, 1), p=(.5, .5), size=N, replace=True)
# Compute the running proportion of heads:
r = np.cumsum(flip_sequence)
n = np.linspace(1, N, N)  # n is a vector.
run_prop = r/n  # component by component division.

# Graph the running proportion:
plt.plot(n, run_prop, '-o', )
plt.xscale('log')  # an alternative to plot() and xscale() is semilogx()
plt.xlim(1, N)
plt.ylim(0, 1)
plt.xlabel('Flip Number')
plt.ylabel('Proportion Heads')
plt.title('Running Proportion of Heads')
# Plot a dotted horizontal line at y=.5, just as a reference line:
plt.axhline(y=.5, ls='dashed')

# Display the beginning of the flip sequence.
flipletters = ''
for i in flip_sequence[:10]:
    if i == 1:
        flipletters += 'H'
    else:
        flipletters += 'T'

plt.text(10, 0.8, 'Flip Sequence = %s...' % flipletters)
# Display the relative frequency at the end of the sequence.
plt.text(25, 0.2, 'End Proportion = %s' % run_prop[-1])

plt.savefig('Figure_3.1.png')
