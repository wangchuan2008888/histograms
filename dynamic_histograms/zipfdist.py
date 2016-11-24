import numpy as np
from scipy import stats

n = 500000
shape = 1.01
loc = 0

z01 = stats.zipf.rvs(a=shape,loc=loc,size=n)
shape += 0.24
z05 = stats.zipf.rvs(a=shape,loc=loc,size=n)
shape += 0.25
z1 = stats.zipf.rvs(a=shape,loc=loc,size=n)
shape += 0.25
z15 = stats.zipf.rvs(a=shape,loc=loc,size=n)
shape += 0.25
z2 = stats.zipf.rvs(a=shape,loc=loc,size=n)
shape += 0.25
z25 = stats.zipf.rvs(a=shape,loc=loc,size=n)
shape += 0.25
z3 = stats.zipf.rvs(a=shape,loc=loc,size=n)

f = open('data/zipfdistributions.csv', 'w')
f.write('zipf1.01,zipf1.25,zipf1.5,zipf1.75,zipf2,zipf2.25,zipf2.5\n')

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
