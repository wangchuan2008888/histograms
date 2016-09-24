import numpy as np
#from pylab import normal, concatenate
#import random
from scipy import stats

n = 1000000
s = 1000

bimodal = np.append(stats.norm.rvs(size=n/2, loc=0, scale=0.2*s), stats.norm.rvs(size=n/2, loc=2*s, scale=0.2*s, ))

f = open('data/biaseddistributions.csv', 'w')
f.write('biasedbimodal,\n')

for i in range(0,n):
    f.write(str(bimodal[i]) + ',' + '\n')
f.close()