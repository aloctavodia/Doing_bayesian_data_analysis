"""
Random height, weight generator for males and females. Uses parameters from
Brainard, J. & Burmaster, D. E. (1992). Bivariate distributions for height and
weight of men and women in the United States. Risk Analysis, 12(2), 267-275.
John K. Kruschke, January 2008.
"""
from __future__ import division
from scipy.stats import multivariate_normal
import numpy as np


def HtWtDataGenerator(nSubj, rndsd=None):
    # Specify parameters of multivariate normal (MVN) distributions.
    # Men:
    HtMmu = 69.18
    HtMsd = 2.87
    lnWtMmu = 5.14
    lnWtMsd = 0.17
    Mrho = 0.42
    Mmean = np.array([HtMmu , lnWtMmu])
    Msigma = np.array([[HtMsd**2, Mrho * HtMsd * lnWtMsd],
                     [Mrho * HtMsd * lnWtMsd, lnWtMsd**2]])

    # Women cluster 1:
    HtFmu1 = 63.11
    HtFsd1 = 2.76
    lnWtFmu1 = 5.06
    lnWtFsd1 = 0.24
    Frho1 = 0.41
    prop1 = 0.46
    Fmean1 = np.array([HtFmu1, lnWtFmu1])
    Fsigma1 = np.array([[HtFsd1**2, Frho1 * HtFsd1 * lnWtFsd1],
                     [Frho1 * HtFsd1 * lnWtFsd1, lnWtFsd1**2]])
    # Women cluster 2:
    HtFmu2 = 64.36
    HtFsd2 = 2.49
    lnWtFmu2 = 4.86
    lnWtFsd2 = 0.14
    Frho2 = 0.44
    prop2 = 1 - prop1
    Fmean2 = np.array([HtFmu2, lnWtFmu2])
    Fsigma2 = np.array([[HtFsd2**2 , Frho2 * HtFsd2 * lnWtFsd2],
                [Frho2 * HtFsd2 * lnWtFsd2 , lnWtFsd2**2]])

    # Randomly generate data values from those MVN distributions.
    if rndsd is not None:
        np.random.seed(rndsd)
    datamatrix = np.zeros((nSubj, 3))
    # arbitrary coding values
    maleval = 1
    femaleval = 0 
    for i in range(0, nSubj):
        # Flip coin to decide sex
        sex = np.random.choice([maleval, femaleval], replace=True, p=(.5,.5), size=1)
        if sex == maleval:
            datum = multivariate_normal.rvs(mean=Mmean, cov=Msigma) 
        if sex == femaleval:
            Fclust = np.random.choice([1, 2], replace=True, p=(prop1, prop2), size=1)
            if Fclust == 1:
                datum = multivariate_normal.rvs(mean=Fmean1, cov=Fsigma1)
            if Fclust == 2: 
                datum = multivariate_normal.rvs(mean=Fmean2, cov=Fsigma2)
        datamatrix[i] = np.concatenate([sex, np.round([datum[0], np.exp(datum[1])], 1)])

    return datamatrix
