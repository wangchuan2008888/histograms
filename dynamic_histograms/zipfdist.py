import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter

c = 50
shape='norm'
deviation = 1
zzipf = 1
szipf = 1
n = 50

z = stats.zipf.rvs(a=zzipf,size=n)
s = stats.zipf.rvs(a=szipf,size=n)

zcount = Counter(z)
scount = Counter(s)




# n = 100000
# scale = 5000
# shape = 1.01
# loc = 0

# z0 = stats.zipf.rvs(a=shape,loc=loc,size=n)
# shape = 1.5
# z05 = stats.zipf.rvs(a=shape,loc=loc,size=n)
# shape = 2
# z1 = stats.zipf.rvs(a=shape,loc=loc,size=n)
# shape = 2.5
# z15 = stats.zipf.rvs(a=shape,loc=loc,size=n)
# shape = 3
# z2 = stats.zipf.rvs(a=shape,loc=loc,size=n)
# shape = 3.5
# z25 = stats.zipf.rvs(a=shape,loc=loc,size=n)
# shape = 4
# z3 = stats.zipf.rvs(a=shape,loc=loc,size=n)

# f = open('data/zipfdistributions.csv', 'w')
# f.write('zipf0.01,zipf0.5,zipf1,zipf1.5,zipf2,zipf2.5,zipf3\n')

# for i in range(0,n):
#     f.write(str(z0[i]) + ',' +
#             str(z05[i]) + ',' +
#             str(z1[i]) + ',' +
#             str(z15[i]) + ',' +
#             str(z2[i]) + ',' +
#             str(z25[i]) + ',' +
#             str(z3[i]) + '\n')
# f.close()
# x = stats.zipf.rvs(a=1.01,size=n)
# print stats.ks_2samp(z0, x)
