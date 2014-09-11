"""
The program described in this section was used to generate Tables 13.1 and 13.2.
The program determines the minimal sample size needed to achieve a specified 
goal with a specified power, when flipping a single coin.
"""
import numpy as np
from HDIofICDF import *
from scipy.special import binom, betaln


def minNforHDIpower(genPriorMean, genPriorN, HDImaxwid=None, nullVal=None,
                    ROPE=None, desiredPower=0.8, audPriorMean=0.5,
                    audPriorN=2, HDImass=0.95, initSampSize=1, verbose=True):
    if HDImaxwid != None  and nullVal != None:
        sys.exit('One and only one of HDImaxwid and nullVal must be specified')
    if ROPE == None:
        ROPE = [nullVal, nullVal]
   # Convert prior mean and N to a, b parameter values of beta distribution.
    genPriorA = genPriorMean * genPriorN
    genPriorB = (1.0 - genPriorMean) * genPriorN
    audPriorA = audPriorMean * audPriorN
    audPriorB = (1.0 - audPriorMean) * audPriorN
    # Initialize loop for incrementing sampleSize
    sampleSize = initSampSize
    # Increment sampleSize until desired power is achieved.
    while True:
        zvec = np.arange(0, sampleSize+1) # All possible z values for N flips.
        # Compute probability of each z value for data-generating prior.
        pzvec = np.exp(np.log(binom(sampleSize, zvec))
                   + betaln(zvec + genPriorA, sampleSize - zvec + genPriorB)
                   - betaln(genPriorA, genPriorB))
        # For each z value, compute HDI. hdiMat is min, max of HDI for each z.
        hdiMat = np.zeros((len(zvec), 2))
        for zIdx in range(0, len(zvec)):
            z = zvec[zIdx]
            # Determine the limits of the highest density interval
            # hdp is a function from PyMC package and takes a sample vector as 
            # input, not a function.
            hdiMat[zIdx] = HDIofICDF(beta, credMass=HDImass, a=(z + audPriorA), 
                                     b=(sampleSize - z + audPriorB))
        if HDImaxwid != None:
            hdiWid = hdiMat[:,1] - hdiMat[:,0]
            powerHDI = np.sum(pzvec[hdiWid < HDImaxwid])
        if nullVal != None:
            powerHDI = np.sum(pzvec[(hdiMat[:,0] > ROPE[1]) | 
                                    (hdiMat[:,1] < ROPE[0])])
        if verbose:
            print " For sample size = %s\npower = %s\n" % (sampleSize, powerHDI)

        if powerHDI > desiredPower:
            break
        else:
            sampleSize += 1
    return sampleSize

print minNforHDIpower(genPriorMean=.85 , genPriorN=2000 , nullVal=0.5, verbose=False)
#print minNforHDIpower(genPriorMean=.85 , genPriorN=10 , HDImaxwid=0.2, verbose=False)

