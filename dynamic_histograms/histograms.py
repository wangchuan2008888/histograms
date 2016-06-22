'''
Given samples, it constructs the appropriate histogram from the sample

Steffani Gomez(smg1)
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Histogram(object):

    # initializes the class with a default number of four buckets
    def __init__(self, frame):
        self.frame = frame
        self.maxlength = len(self.frame.index)
        self.numbuckets = 5
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0, 
                'frequency': 0,
                'size': 0
            })
        self.buckets = buckets
        self.minimum = 0
        self.maximum = 0

    # determines the frequency of 
    def frequency_over_range(attr, sample, low, high):
        frequency = 0
        for val in list(sample[attr]):
            if low <= val and val < high:
                frequency += 1
        return frequency

    # creates the initial histogram from the sample on the atttribute, using only the sample's min and max
    # since the intial self-tuning histogram does not look at the data and assumes a frequency of maximum 
    # observations / # of buckets for each bucket
    def create_initial_sf_histogram(self, attr, sample):
        minimum = min(list(sample[attr]))
        maximum = max(list(sample[attr]))
        self.minimum = minimum
        self.maximum = maximum
        range = math.ceil(maximum - minimum) # want to make sure we capture the maximum element in the last bucket
        #if (range % self.numbuckets == 0):
        size = math.ceil(range / self.numbuckets)
        low = minimum
        high = minimum + size
        for bucket in self.buckets:
            bucket['low'] = low
            bucket['high'] = high
            bucket['frequency'] = round(self.maxlength / self.numbuckets)#frequency_over_range(attr, sample, low, high)
            bucket['size'] = size
            low = high
            high += size

    # plots a histogram via matplot.pyplot. this is the intial histogram of the self-tuning histogram which is both equi-depth
    # and equi-width (because the intial histogram does not look at the data frequencies)
    def plot_initial_sf_histogram(self):
        buckets = []
        frequency = []
        size = 0
        for bucket in self.buckets:
            buckets.append(bucket['low'])
            frequency.append(bucket['frequency'])
            size = bucket['size']
        print buckets
        print frequency
        print self.minimum
        print self.maximum
        #plt.hist(frequency, range=(self.minimum, self.maximum), bins=buckets, weights=frequency)
        plt.bar(buckets, frequency, width=size)
        plt.grid(True)
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([0, frequency[0] + (frequency[0] / 2)])
        #plt.axis([self.minimum - size, self.maximum + size, 0, frequency[0] + (frequency[0] / 2)])
        plt.show()

        # add a 'best fit' line
        # y = mlab.normpdf( bins, mu, sigma)
        #l = plt.plot(bins)

        #plt.xlabel('Smarts')
        #plt.ylabel('Probability')
        #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
        #plt.axis([40, 160, 0, 0.03])
        #plt.grid(True)

        #plt.show()

