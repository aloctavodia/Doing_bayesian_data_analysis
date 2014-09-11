"""
This program finds the HDI of a probability density function that is specified 
mathematically in Python.
"""
from scipy.optimize import fmin
from scipy.stats import *


def HDIofICDF(dist_name, credMass=0.95, **args):
    distri = dist_name(**args)
    #distri = norm(0, 1)
    incredMass =  1.0 - credMass
    def intervalWidth(lowTailPr):
        return distri.ppf(credMass + lowTailPr) - distri.ppf(lowTailPr)

    optInfo = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)
    HDIlowTailPr = optInfo
    return np.array((distri.ppf(HDIlowTailPr)[0], 
                    distri.ppf(credMass + HDIlowTailPr)[0]))
