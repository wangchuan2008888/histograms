"""
It constructs an equi-width histogram from the dataset given.

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
from collections import Counter
import user_distribution
import json
from scipy import stats

class Control_Histogram(object):

    """
    This class models an instance of a control histogram, which is equi-width and stretches 
    its bucket boundaries to include values that are beyond the leftmost and rightmost buckets.
    """

    def __init__(self, file, numbuckets, outputpath):
        
        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """
        self.outputpath = outputpath
        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'size': 0,
                'frequency': 0
            })
        self.buckets = buckets
        self.counter = 0
        self.min = float('inf')
        self.max= float('-inf')

    def create_histogram(self, attr, batchsize, userbucketsize):
        """Reads through the file and creates the histogram, adding in data points as they are being read."""
        N = 0
        sample = []
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
                    self.create_initial_histogram(N, set(sample))
                    self.plot_histogram(attr, self.buckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    initial = True
                elif initial == True:
                    self.add_datapoint(float(row[attr_index]))
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
            #self.print_buckets()
            x = stats.zipf.rvs(a=1.01,size=100000)
            print stats.ks_2samp(realdist, x)

            # we could be having issues due to the fact that we need the cdf of a discrete distribution not continuous which is probably different
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
        plt.savefig(self.outputpath + "//img//controlcdf" + str(self.counter) + ".jpg")
        self.counter += 1
        plt.close()

    def callable_cdf(self, x, cumfreq):
        values = []
        for value in x:
            v = self.cdf(value, cumfreq)
            if v == None:
                #self.print_buckets()
                print '{0:.16f}'.format(self.buckets[self.numbuckets - 1]['high'])
                print value, v
                print '{0:.16f}'.format(value)
                print '{0:.16f}'.format(self.max)
                print self.min, self.max
                if value > self.min:
                    print "here"
                    v = 1
                else:
                    print "or here"
                    v = 0
                print value, v
            values.append(v)
        return np.array(values)

    def callable_linearcdf(self, x, cumfreq):
        values = []
        for value in x:
            values.append(self.linear_cdf(value, cumfreq))
        return np.array(values)
    
    def cdf(self, x, cumfreq):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                return cumfreq[i] / cumfreq[len(cumfreq) - 1]

    def linear_cdf(self, x, cumfreq):
        if x <= self.min:
            return 0
        elif x >= self.max:
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

    def create_initial_histogram(self, N, sample):
        """Creates the bucket boundaries based on the first n distinct points present in the sample."""
        sample = list(sample)
        sorted_sample = sorted(sample, key=float)
        c = Counter(sorted_sample)
        r = max(sorted_sample) - min(sorted_sample)
        width = r / self.numbuckets
        low = sorted_sample[0]
        for i in range(0, self.numbuckets):
            self.buckets[i]['size'] = width
            self.buckets[i]['low'] = low
            self.buckets[i]['high'] = low + width
            for j in range(0, len(sorted_sample)):
                if sorted_sample[j] >= low and sorted_sample[j] < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += c[sorted_sample[j]]
                elif sorted_sample[j] >= self.buckets[i]['high']:
                    break
            low = self.buckets[i]['high']
        self.buckets[0]['low'] = self.min
        self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
    
    def add_datapoint(self, value):
        """Increases the count of the bucket the value belongs in the histogram."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        elif value >= self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1

    def plot_histogram(self, attr, buckets):
        """It plots the histogram. """
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
        plt.title(r'$\mathrm{Control\ Histogram\ of\ ' + attr + '}$')

        with open(self.outputpath + "//data//control" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//control" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
