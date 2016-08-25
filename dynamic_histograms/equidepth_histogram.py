"""
It constructs an equi-depth histogram from the dataset given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
import random
import user_distribution
import json
from scipy import stats

upper_factor = 3

class Equidepth_Histogram(object):

    """
    This class models an instance of an equi-depth histogram, which is a histogram with all the buckets have the same count.
    """

    def __init__(self, f, numbuckets, outputpath):

        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """
        self.outputpath = outputpath
        self.file = f
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, numbuckets):
            buckets.append({
                'low': 0,
                'high': 0, 
                'frequency': 0,
                'size': 0,
            })
        self.buckets = buckets
        self.threshold = None
        self.counter = 0
        self.min = float('inf')
        self.max= float('-inf')
        self.upper = numbuckets * upper_factor

    def create_histogram(self, attr, l, batchsize, userbucketsize):
        """l is a tunable parameter (> -1) which influences the upper thresholder of bucket count for all buckets. The appropriate bucket counter is 
        incremented for every record read in. If a bucket counter reaches threshold, the bucket boundaries are recalculated and the threshold is updated."""
        N = 0
        sample = []
        skipcounter = 0
        skip = 0
        initial = False
        with open(self.file) as f:
            reader = csv.reader(f)
            header = reader.next()
            for i in range(0, len(header)):
                header[i] = unicode(header[i], 'utf-8-sig')
            attr_index = header.index(attr)
            for row in reader:
                N += 1
                if float(row[attr_index]) < self.min:
                    self.min = float(row[attr_index])
                if float(row[attr_index]) > self.max:
                    self.max = float(row[attr_index]) 
                if len(set(sample)) < self.numbuckets:
                    sample.append(float(row[attr_index]))
                elif len(set(sample)) == self.numbuckets and initial == False:
                    self.create_initial_histogram(N, sample, l)
                    self.plot_histogram(attr, self.buckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    skipcounter += 1
                    self.add_datapoint(float(row[attr_index]), N, sample, attr, l)
                    if skipcounter == skip:
                        sample = self.maintainBackingSample(float(row[attr_index]), sample)
                        skip = self.calculateSkip(len(sample))
                        skipcounter = 0
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_histogram(attr, self.buckets)
                        d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        d.create_distribution(self.buckets)
                        new_buckets = d.return_distribution()
                        self.plot_histogram(attr, new_buckets)
                        self.compare_histogram(attr, False)
        self.compare_histogram(attr, True)

    def compare_histogram(self, attr, end):
        frequency = []
        binedges = []
        for bucket in self.buckets:
            frequency.append(bucket['frequency'])
            binedges.append(bucket['low'])
        binedges.append(bucket['high'])
        cumfreq = np.cumsum(frequency)
        realdist = np.array(pd.read_csv(self.file)[attr], dtype=float)
        if end:
            print stats.kstest(realdist, lambda x: self.callable_cdf(x, cumfreq), N=len(realdist), alternative='two-sided')
            print stats.kstest(realdist, lambda x: self.callable_linearcdf(x, cumfreq), N=len(realdist), alternative='two-sided')
        sorted_data = np.sort(realdist)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.grid(True)
        plt.plot(sorted_data, yvals)
        step = [0]
        step.extend(cumfreq / cumfreq[len(cumfreq) - 1])
        plt.step(binedges[0:], step)
        plt.plot(binedges[0:], step)
        plt.legend(['CDF of real data', 'CDF of histogram', 'CDF of linear approx'], loc='lower right')
        plt.savefig(self.outputpath + "//img//equidepthcdf" + str(self.counter) + ".jpg")  
        self.counter += 1
        plt.close()
        
    def callable_cdf(self, x, cumfreq):
        values = []
        for value in x:
            v = self.cdf(value, cumfreq)
            if v == None:
                print value, v
                print self.min, self.max
            values.append(v)
        return np.array(values)

    def callable_linearcdf(self, x, cumfreq):
        values = []
        for value in x:
            values.append(self.linear_cdf(value, cumfreq))
        return np.array(values)
    
    def cdf(self, x, cumfreq):
        if x < self.min:
            return 0
        elif x > self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                return cumfreq[i] / cumfreq[len(cumfreq) - 1]

    def linear_cdf(self, x, cumfreq):
        if x < self.min:
            return 0
        elif x > self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                approx = None
                percentage = (x - self.buckets[i]['low']) / self.buckets[i]['size']
                if i > 0:
                    approx = percentage + cumfreq[i - 1]
                else:
                    approx = percentage * cumfreq[i]
                return approx / cumfreq[len(cumfreq) - 1]   

    def create_initial_histogram(self, N, sample, l):
        """Creates the initial histogram boundaries from the first n distinct values and sets the threshold along with l (lambda)."""
        sorted_sample = sorted(sample, key=float)
        for i in range(0, self.numbuckets):
            self.buckets[i]['low'] = sorted_sample[i]
            if i == self.numbuckets - 1:
                self.buckets[i]['high'] = sorted_sample[i] + 1
            else:
                self.buckets[i]['high'] = sorted_sample[i + 1]
            self.buckets[i]['frequency'] = N / self.numbuckets
            self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
        self.buckets[0]['low'] = self.min
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.threshold = (2 + l) * (N / self.numbuckets)

    # since the pseudocode doesn't mention what to do about values that fall outside of buckets, I extend those bucket
    # boundaries to include values that are not included in the boundary range 
    def add_datapoint(self, value, N, sample, attr, l):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - value
            if self.buckets[0]['frequency'] >= self.threshold:
                self.thresholdReached(self.buckets[0], N, sample, attr, l)
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = value + 1 - self.buckets[self.numbuckets - 1]['low']
            if self.buckets[self.numbuckets - 1]['frequency'] >= self.threshold:
                self.thresholdReached(self.buckets[self.numbuckets - 1], N, sample, attr, l)
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    if self.buckets[i]['frequency'] >= self.threshold:
                        self.thresholdReached(self.buckets[i], N, sample, attr, l)

    def calculateSkip(self, n):
        v = random.uniform(0, 1)
        l = 0
        t = n + 1
        num = 1
        quot = num / t
        while quot > v:
            l += 1
            t += 1
            num += 1
            quot = (quot * num) / t
        return l

    def maintainBackingSample(self, value, sample):
        if len(sample) + 1 <= self.upper:
            sample.append(value)
        else:
            rand_index = random.randint(0,len(sample) - 1)
            sample[rand_index] = value
        return sample

    def mergebucketPair(self):
        minimum = float('-inf')
        index = None
        for i in range(0, self.numbuckets - 1):
            if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.threshold:
                if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < minimum:
                    minimum = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
                    index = i
        return index


    def thresholdReached(self, bucket, N, sample, attr, l):
        sample = list(sample)
        index = self.mergebucketPair()
        if index != None:
            bucket2 = self.buckets[index]
            self.mergebuckets(self.buckets[i], self.buckets[i + 1])
            self.splitbucket(bucket, bucket['low'], bucket['high'], sample)
        else:
            self.computehistogram(sample, N)
            self.threshold = (2 + l) * (N / self.numbuckets)

    def computehistogram(self, sample, N):
        sample = list(sample)
        sample = sorted(sample, key=float)
        frac = len(sample) / self.numbuckets
        equal = N / self.numbuckets
        for i in range(1, self.numbuckets):
            index = int(round(i * frac))
            self.buckets[i]['low'] = sample[index]
            if i == self.numbuckets - 1:
                self.buckets[i]['high'] = sample[index] + 1
            else:
                self.buckets[i]['high'] = sample[int(round((i + 1) * frac))]
            self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
            self.buckets[i]['frequency'] = (i * equal) - ((i - 1) * equal)
        self.buckets[0]['low'] = self.min
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1 

    def mergebuckets(self, bucket1, bucket2):
        """Merging two buckets into one bucket in the list of buckets."""
        mergedbucket = {
            'low': bucket1['low'],
            'high': bucket2['high'],
            'size': bucket2['high'] - bucket1['low'],
            'frequency': bucket1['frequency'] + bucket2['frequency']
        }
        buckets = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]['low'] == bucket1['low'] and self.buckets[i]['high'] == bucket1['high']:
                buckets.append(mergedbucket)
            if self.buckets[i]['low'] == bucket2['low'] and self.buckets[i]['high'] == bucket2['high']:
                pass
            else:
                buckets.append(self.buckets[i])
        self.buckets = buckets

    def splitbucket(self, bucket, low, high, sample):
        """Splits a bucket in the list of buckets of the histogram."""
        s = []
        for i in range(0, len(sample)):
            if sample[i] >= low and sample[i] < high:
                s.append(sample[i])
        m = np.median(s)
        b = {
            'low': bucket['low'],
            'high': sample[int(len(s) // 2)],
            'size': sample[int(len(s) // 2)] - bucket['low'],
            'frequency': self.threshold // 2
        }
        bucket['low'] = sample[int(len(s) // 2) + 1]
        bucket['high'] = high
        bucket['frequency'] = self.threshold // 2
        bucket['size'] = high - sample[int(len(s) // 2) + 1]
        buckets = []
        for i in range(0, len(self.buckets)):
            if self.buckets[i]['low'] == bucket['low'] and self.buckets[i]['high'] == bucket['high']:
                buckets.append(b)
                buckets.append(bucket)
            else:
                buckets.append(self.buckets[i])
        self.buckets = buckets

    def plot_histogram(self, attr, buckets):
        """Plots the histogram."""
        bins = []
        frequency = []
        for bucket in buckets:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
        bins.append(bucket['high'])

        frequency = np.array(frequency)
        bins = np.array(bins)

        widths = bins[1:] - bins[:-1]

        plt.bar(bins[:-1], frequency, width=widths, edgecolor=['black'], color='#348ABD')

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim(self.min - abs(buckets[0]['size']), self.max + abs(buckets[0]['size']))
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.subplot().set_axis_bgcolor('#E5E5E5');
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Equi-Depth\ Histogram\ of\ ' + attr + '}$')
                
        with open(self.outputpath + "//data//equidepth" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//equidepth" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""        
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"