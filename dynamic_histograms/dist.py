import matplotlib.pyplot as plt
import numpy as np
from pylab import normal, concatenate
import random
from scipy import stats

n = 1000000
s = 1000

#norm = np.random.randn(n) * s
norm = stats.norm.rvs(size=n, scale=s)
#chi = np.random.chisquare(4,  n) * s
chi = stats.chi2.rvs(size=n, df=4.0, scale=s)
#logistic = np.random.logistic(10, 1, n) * s
loc = 10
scale = 1
logistic = stats.logistic.rvs(size=n, loc=loc, scale=s)
#beta = np.random.beta(0.5, 0.5, n) * s
a = 0.5
b = 0.5
beta = stats.beta.rvs(size=n, a=a, b=b, scale=s)
#gamma = np.random.gamma(7, 0.5, n) * s
shape = 7
#scale = 0.5
gamma = stats.gamma.rvs(size=n, a=shape, scale=0.5 * s)
#bimodal = concatenate((normal(1,.2,n*0.75),normal(2,.2,n*0.25))) * s
bimodal = concatenate((stats.norm.rvs(1, .2 * s,n*0.75), stats.norm.rvs(2,.2 * s,n*0.25)))
if len(bimodal) != n:
    x = stats.norm.rvs(2, .2 * s, 1)
    #bimodal = concatenate(bimodal, x)
    bimodal = np.append(bimodal, x)
#random.shuffle(bimodal)
#uniform = np.random.uniform(0, s, n)
loc = 0
uniform = stats.uniform.rvs(size=n, loc=loc, scale=s)

f = open('data/distributions.csv', 'w')
f.write('norm,chi,logistic,beta,gamma,bimodal,uniform\n')

for i in range(0,n):
    f.write(str(norm[i]) + ',' +
            str(chi[i]) + ',' +
            str(logistic[i]) + ',' +
            str(beta[i]) + ',' +
            str(gamma[i]) + ',' +
            str(bimodal[i]) + ',' +
            str(uniform[i]) + '\n')
f.close()