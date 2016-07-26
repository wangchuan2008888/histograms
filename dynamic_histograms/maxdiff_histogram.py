"""
It constructs a simple dynamic max-diff histogram rom the dataset given.

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
from collections import defaultdict
import operator
import itertools
import random

class MaxDiff_Histogram(object):

    """
    This class models an instance of a max-diff histogram histogram, which is a histogram that sets boundaries on the
    numbuckets - 1 largest differences in area between the values.
    """

    def __init__(self, file, numbuckets):

        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """

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


    def create_histogram(self, attr, batchsize):
        """Reads in records from the file, computing the initial histogram and after each batch by finding numbuckets - 1 
        largest differences in area between each value in the sample and setting the boundaries in between these values."""
        N = 0
        sample = []
        initial = False
        skip = 0
        skipcounter = 0
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
                if len(set(sample)) < self.numbuckets * 2:
                    sample.append(float(row[attr_index]))
                elif len(set(sample)) == self.numbuckets * 2 and initial == False:
                    self.compute_histogram(sample, N)
                    self.plot_histogram(attr)
                    skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    skipcounter += 1
                    self.add_datapoint(float(row[attr_index]))
                    if skipcounter == skip:
                        sample = self.maintainBackingSample(float(row[attr_index]), sample, self.numbuckets)
                        skip = self.calculateSkip(len(sample))
                        skipcounter = 0
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_histogram(attr)
                        self.compute_histogram(sample, N)
                        self.plot_histogram(attr)

    def compute_histogram(self, sample, N):
        """Computes the histogram boundaries by finding the numbuckets - 1 largest differences in areas 
        between values in the sample and then arranges the buckets to have the proper frequency."""
        sorted_sample = sorted(sample, key=float)
        c = Counter(sorted_sample)
        a = [-1] * (self.numbuckets - 1)
        bucketarea = defaultdict(list)
        for i in range(0, len(sorted_sample) - 2):
            freq = c[sorted_sample[i]]
            spread = sorted_sample[i + 1] - sorted_sample[i]
            area = spread * freq
            area1 = (sorted_sample[i + 2] - sorted_sample[i + 1]) * c[sorted_sample[i + 1]]
            delta = abs(area1 - area)
            if self.checkDifference(delta, a):
                stats = self.addArea(delta, a, bucketarea, i)
                a = stats[0]
                bucketarea = stats[1]
        self.arrangeBuckets(c, a, bucketarea, sorted_sample, N)

    def addArea(self, area, a, bucketarea, index):
        """Adds the area to the list of areas (a) and the dictionary of areas (bucketarea) and the index 
        of the value."""
        length = len(a)
        for i in range(0, len(a)):
            if area >= a[i]:
                bucketarea[area].append(index)
                a[i:i] = [area]
                area = -2
        val = sorted(a, reverse=True)[length:][0]
        final = sorted(a, reverse=True)[0:length]
        if bucketarea.has_key(val):
            if val not in final:
                del bucketarea[val]
            else:
                l = len(bucketarea[val])
                bucketarea[val] = bucketarea[val][:l - 1]
        return final, bucketarea

    def checkDifference(self, area, a):
        """Checks whether the area is greater than any of the areas in the list (a)."""
        for i in range(0, len(a)):
            if area >= a[i]:
                return True
        return False

    def arrangeBuckets(self, counter, areas, bucketarea, sample, N):
        """Arranges the bucket in order by setting the boundaries in order and calculates the 
        frequency for each bucket."""
        boundaries = sorted(bucketarea.items(), key=operator.itemgetter(1))
        low = self.min
        values = bucketarea.values()
        values = list(itertools.chain(*values))
        values = sorted(values)
        print values
        print counter
        print sample
        for i in range(0, len(values)):
            self.buckets[i]['low'] = low
            highindex = values[i]
            self.buckets[i]['high'] = sample[highindex]
            self.buckets[i]['size'] = sample[highindex] - low
            if sample[highindex] == self.buckets[i]['low']:
                self.buckets[i]['high'] = sample[highindex + 1]
                self.buckets[i]['size'] = sample[highindex + 1] - low
            if low == self.min:
                self.buckets[i]['frequency'] = counter[sample[0]] * N / len(sample)
            else:
                self.buckets[i]['frequency'] = counter[low] * N / len(sample)
            low = self.buckets[i]['high']
        print "index: " + str(i)
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.buckets[self.numbuckets - 1]['low'] = self.buckets[self.numbuckets - 2]['high']
        self.buckets[self.numbuckets - 1]['frequency'] = counter[self.buckets[self.numbuckets - 1]['low']] * N / len(sample)
        self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        #self.buckets[self.numbuckets - 1]['frequency'] = counter[self.buckets[self.numbuckets - 1]['low']] * N / len(sample)
        #self.buckets[i]['high'] = self.max + 1
        #self.buckets[i]['size'] = self.max + 1 - self.buckets[self.numbuckets - 1]['low']
        #index = 0
        #bucket = self.buckets[index]
        #for i in range(0, len(sample)):
        #    if sample[i] >= bucket['high']:
        #        index += 1
        #        bucket = self.buckets[index]
        #    if sample[i] >= bucket['low'] and sample[i] < bucket['high']:
        #        bucket['frequency'] += counter[sample[i]]

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

    def maintainBackingSample(self, value, sample, upper):
        if len(sample) + 1 <= upper:
            sample.append(value)
        else:
            rand_index = random.randint(0,len(sample) - 1)
            sample[rand_index] = value
        return sample

    def add_datapoint(self, value):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1

    def plot_histogram(self, attr):
        """Plots the histogram."""
        self.print_buckets()
        bins = []
        frequency = []
        for i in range(0, self.numbuckets):
            bins.append(self.buckets[i]['low'])
            frequency.append(self.buckets[i]['frequency'])
        bins.append(self.buckets[i]['high'])

        frequency = np.array(frequency)
        bins = np.array(bins)

        widths = bins[1:] - bins[:-1]

        print bins
        print bins[:-1]
        print frequency
        print widths

        plt.bar(bins[:-1], frequency, width=widths)

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([self.min * 1.5, self.max * 1.5])
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Max-Diff\ Histogram\ of\ ' + attr + '}$')
        path = "maxdiff" + str(self.counter) + ".jpg"
        plt.savefig(path)
        plt.clf()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"