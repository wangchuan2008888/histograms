from __future__ import division
import numpy as np
from scipy import stats

n = 50000
shape = 1.01
loc = 0

z01 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z01)
maximum = max(z01)
z01 = (z01 - minimum)*(10000/(maximum-minimum))
shape += 0.04
z05 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z05)
maximum = max(z05)
z05 = (z05 - minimum)*(10000/(maximum-minimum))
shape += 0.05
z1 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z1)
maximum = max(z1)
z1 = (z1 - minimum)*(10000/(maximum-minimum))
shape += 0.05
z15 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z15)
maximum = max(z15)
z15 = (z15 - minimum)*(10000/(maximum-minimum))
shape += 0.05
z2 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z2)
maximum = max(z2)
z2 = (z2 - minimum)*(10000/(maximum-minimum))
shape += 0.05
z25 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z25)
maximum = max(z25)
z25 = (z25 - minimum)*(10000/(maximum-minimum))
shape += 0.05
z3 = stats.zipf.rvs(a=shape,loc=loc,size=n)
minimum = min(z3)
maximum = max(z3)
z3 = (z3 - minimum)*(10000/(maximum-minimum))

f = open('data/zipfdistributions.csv', 'w')
f.write('zipf1.01,zipf1.05,zipf1.1,zipf1.15,zipf1.2,zipf1.25,zipf1.3\n')

for i in range(0,n):
    f.write(str(z01[i]) + ',' +
            str(z05[i]) + ',' +
            str(z1[i]) + ',' +
            str(z15[i]) + ',' +
            str(z2[i]) + ',' +
            str(z25[i]) + ',' +
            str(z3[i]) + '\n')
f.close()
#x = stats.zipf.rvs(a=1.01,size=n)
#print stats.ks_2samp(z0, x)
