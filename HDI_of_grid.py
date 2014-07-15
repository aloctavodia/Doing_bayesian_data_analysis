"""
Arguments:
probMassVec is a vector of probability masses at each grid point.
credMass is the desired mass of the HDI region.

Return a dictionary with:
indices is a vector of indices that are in the HDI
mass is the total mass of the included indices
height is the smallest component probability mass in the HDI
"""
import numpy as np

def HDI_of_grid(probMassVec, credMass=0.95):
    sortedProbMass = np.sort(probMassVec, axis=None)[::-1]
    HDIheightIdx = np.min(np.where(np.cumsum(sortedProbMass) >= credMass))
    HDIheight = sortedProbMass[HDIheightIdx]
    HDImass = np.sum(probMassVec[probMassVec >= HDIheight])
    idx = np.where(probMassVec >= HDIheight)
    return {'indices':idx, 'mass':HDImass, 'height':HDIheight}

if  __name__ =='__main__':
    from scipy.stats import beta
    theta1 = np.linspace(0, 1, 10)
    theta2 = theta1
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)
    probDensityVec = beta.pdf(theta1_grid, 3, 3)
    probMassVec = probDensityVec / np.sum(probDensityVec)
    HDIinfo = HDIofGrid(probMassVec)
    print HDIinfo
