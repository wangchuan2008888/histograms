import matplotlib.pyplot as plt
import numpy as np
from pylab import normal, concatenate
import random

n = 1000000
#n = 15000
s = 1000

norm = np.random.randn(n) * s
chi = np.random.chisquare(4,  n) * s
logistic = np.random.logistic(10, 1, n) * s
beta = np.random.beta(0.5, 0.5, n) * s
gamma = np.random.gamma(7, 0.5, n) * s
bimodal = concatenate((normal(1,.2,n*0.75),normal(2,.2,n*0.25))) * s
random.shuffle(bimodal)
uniform = np.random.uniform(0, s, n)

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