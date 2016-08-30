import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

n = 100000
scale = 5000
shape = 1.01
loc = 0

z = stats.zipf.rvs(a=shape,loc=loc,size=n)

f = open('data/zipfdistribution.csv', 'w')
f.write('zipf,\n')

for i in range(0,n):
    f.write(str(z[i]) + ',\n')
f.close()
realdist = np.array(pd.read_csv('data/zipfdistribution.csv')['zipf'], dtype=float)
x = stats.zipf.rvs(a=shape,loc=loc,size=n)
print stats.ks_2samp(realdist, x) #stats.kstest(realdist, 'zipf', args=(shape,loc,n))
