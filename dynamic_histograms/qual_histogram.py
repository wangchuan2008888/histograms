from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

bins = [-3, 2, 5, 10]
frequency = [5, 7, 3]

frequency = np.array(frequency)
bins = np.array(bins)

widths = bins[1:] - bins[:-1]

plt.bar(bins[:-1], frequency, width=widths, edgecolor=['black'], color='#348ABD')

plt.grid(True)
axes = plt.gca()
axes.set_xlim([-5, 12])
axes.set_ylim([0, 10])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(r'$\mathrm{Qualification\ Histogram}$')
path = "qualification.jpg"
plt.savefig(path)
plt.clf()
