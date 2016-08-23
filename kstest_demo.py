from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats

# SEE IF YOU CAN PLOT THE DISTRIBUTIONS AGAINST EACH OTHER, LIKE PLOTTING THE CDF'S OF BOTH DISTRIBUTIONS ON THE SAME GRAPH
# it also seems that the distribution is drawn from distributions in the numpy library and we're testing against distributions 
# in the scipy stats module

realdist = np.array(pd.read_csv('dynamic_histograms/data/testdistributions.csv')['norm'], dtype=float)
ksstats = stats.kstest(realdist, 'norm')
print "ks test statistic for comparing raw data to normal distribution with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
ksstats = stats.kstest(stats.norm.rvs(size=20), 'norm')
print "ks test statistic comparing norm vs norm with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
norm = stats.norm.rvs(size=20)
kssample = stats.ks_2samp(realdist, norm)
print "ks test statistic comparing numpy norm and stats norm with n = 20: " + str(kssample[0]) + ", " + str(kssample[1])
print ""

realdist = np.array(pd.read_csv('dynamic_histograms/data/testdistributions.csv')['chi'], dtype=float)
df = 4.0
size = 20
ksstats = stats.kstest(realdist, 'chi2', args=(df,))
print "ks test statistic for comparing raw data to chi distribution with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
chi = stats.chi2.rvs(size=20, df=4.0)
ksstats = stats.kstest(chi, 'chi2', args=(df,))
print "ks test statistic comparing chi vs chi with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
kssample = stats.ks_2samp(realdist, chi)
print "ks test statistic comparing numpy chisquared and stats chisquared with n = 20: " + str(kssample[0]) + ", " + str(kssample[1])
print ""

realdist = np.array(pd.read_csv('dynamic_histograms/data/testdistributions.csv')['logistic'], dtype=float)
loc = 10.0
scale = 1.0
ksstats = stats.kstest(realdist, 'logistic', args=(loc, scale))
print "ks test statistic for logistic distribution with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
logistic = stats.logistic.rvs(size=20, loc=loc, scale=scale)
ksstats = stats.kstest(logistic, 'logistic', args=(loc, scale))
print "ks test statistic comparing logistic vs logistic with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
kssample = stats.ks_2samp(realdist, logistic)
print "ks test statistic comparing numpy logistic and stats logistic with n = 20: " + str(kssample[0]) + ", " + str(kssample[1])
print ""

realdist = np.array(pd.read_csv('dynamic_histograms/data/testdistributions.csv')['beta'], dtype=float)
a = 0.5
b = 0.5
ksstats = stats.kstest(realdist, 'beta', args=(a, b))
print "ks test statistic for beta distribution with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
beta = stats.beta.rvs(size=20, a=a, b=b)
ksstats = stats.kstest(beta, 'beta', args=(a, b))
print "ks test statistic comparing beta vs beta with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
kssample = stats.ks_2samp(realdist, beta)
print "ks test statistic comparing numpy beta and stats beta with n = 20: " + str(kssample[0]) + ", " + str(kssample[1])
print ""

realdist = np.array(pd.read_csv('dynamic_histograms/data/testdistributions.csv')['gamma'], dtype=float)
shape = 7
scale = 0.5
ksstats = stats.kstest(realdist, 'gamma', args=(shape, 0, scale))
print "ks test statistic for gamma distribution with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
gamma = stats.gamma.rvs(size=20, a=shape, scale=scale)
ksstats = stats.kstest(gamma, 'gamma', args=(shape, 0, scale))
print "ks test statistic comparing gamma vs gamma with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
kssample = stats.ks_2samp(realdist, gamma)
print "ks test statistic comparing numpy gamma and stats gamma with n = 20: " + str(kssample[0]) + ", " + str(kssample[1])
print ""

realdist = np.array(pd.read_csv('dynamic_histograms/data/testdistributions.csv')['uniform'], dtype=float)
loc = 0.0
scale = 1000
ksstats = stats.kstest(realdist, 'uniform', args=(loc, scale))
print "ks test statistic for uniform distribution with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
uniform = stats.uniform.rvs(size=20, loc=loc, scale=scale)
ksstats = stats.kstest(uniform, 'uniform', args=(loc, scale))
print "ks test statistic comparing uniform vs uniform with n = 20: " + str(ksstats[0]) + ", " + str(ksstats[1])
kssample = stats.ks_2samp(realdist, uniform)
print "ks test statistic comparing numpy uniform and stats uniform with n = 20: " + str(kssample[0]) + ", " + str(kssample[1])
