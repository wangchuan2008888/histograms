'''
Given samples, it constructs the appropriate histogram from the sample

Steffani Gomez(smg1)
'''

from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
from collections import Counter

class DVO_Histogram(object):

    def __init__(self, file, numbuckets):
        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'size': 0,
                'leftcounter': 0,
                'rightcounter': 0
            })
        self.buckets = buckets

    def plot_dvo_histogram(self, attr):
        bins = []
        frequency = []
        for bucket in self.buckets:
            bins.append(bucket['low'])
            frequency.append(bucket['leftcounter'])
            bins.append(bucket['low'] + (bucket['size'] / 2))
            frequency.append(bucket['rightcounter'])
        bins.append(bucket['high'])

        frequency = np.array(frequency)
        bins = np.array(bins)

        widths = bins[1:] - bins[:-1]

        plt.bar(bins[:-1], frequency, width=widths, edgecolor=['black'])

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([self.buckets[0]['low'] - self.buckets[0]['size'], self.buckets[self.numbuckets - 1]['high'] * 1.5])
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Histogram\ of\ ' + attr + '}$')
        plt.show()

    # implements a dynamic v-optimal histogram while reading the file
    def create_dvo_histogram(self, attr, batchsize):
         N = 0
         sample = []
         with open(self.file) as f:
            reader = csv.reader(f)
            header = reader.next()
            for i in range(0, len(header)):
                header[i] = unicode(header[i], 'utf-8-sig')
            attr_index = header.index(attr)
            for row in reader:
                sample.append(float(row[attr_index]))
                N += 1
                if len(set(sample)) == self.numbuckets:
                    sorted_sample = sorted(sample, key=float)
                    buckets = sorted(list(set(sample)), key=float)
                    c = Counter(sorted_sample)
                    for i in range(0, self.numbuckets):
                        self.buckets[i]['low'] = buckets[i]
                        if i == self.numbuckets - 1:
                            self.buckets[i]['high'] = buckets[i] + 1
                        else:
                            self.buckets[i]['high'] = buckets[i + 1]
                        self.buckets[i]['leftcounter'] = c[buckets[i]]
                        self.buckets[i]['rightcounter'] = c[buckets[i]]
                        self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                elif len(set(sample)) > self.numbuckets:
                    self.add_datapoint(float(row[attr_index]))
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_dvo_histogram(attr)

    # this method adds data points to the histograms, adjusting the end bucket partitions if necessary
    # WHAT ABOUT IF THE VALUE IS LESS THAN THE LEFTMOST BUCKET BOUNDARY????
    def add_datapoint(self, value):
        if value > self.buckets[self.numbuckets - 1]['high']:
            bucket = {
                'low': self.buckets[self.numbuckets - 1]['high'],
                'high': value + 1,
                'leftcounter': 1,
                'rightcounter': 1,
                'size': value + 1 - self.buckets[self.numbuckets - 1]['high']
            } # borrow one bucket
            index = self.findBestToMerge()[1]
            self.mergebuckets(self.buckets[index], self.buckets[index + 1])
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    if value < self.buckets[i]['low'] + (self.buckets[i]['size'] / 2):
                        self.buckets[i]['leftcounter'] += 1
                    else:
                        self.buckets[i]['rightcounter'] += 1
            s = self.buckets[self.findBestToSplit()[1]]
            mindex = self.findBestToMerge()[1]
            if self.bucketError(s) > self.adjacentbucketsError(self.buckets[mindex], self.buckets[mindex + 1]):
                # split s
                # merge m and m.next
                self.splitbucket(s)
                self.mergebuckets(self.buckets[mindex], self.buckets[mindex + 1])

    # merging two buckets into one bucket in the list of buckets
    def mergebuckets(self, bucket1, bucket2):
        mergedbucket = {
            'low': bucket1['low'],
            'high': bucket2['high'],
            'size': bucket2['high'] - bucket1['low'],
            'leftcounter': bucket1['leftcounter'] + bucket1['rightcounter'],
            'rightcounter': bucket2['leftcounter'] + bucket2['rightcounter']
        }
        buckets = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]['low'] < bucket1['low'] or self.buckets[i]['low'] >= bucket2['high']:
                buckets.append(self.buckets[i])
            elif self.buckets[i]['low'] == bucket1['low']:
                buckets.append(mergedbucket)
            elif self.buckets[i]['low'] == bucket2['low']:
                pass

    def splitbucket(self, bucket):
        bucket1 = {
            'low': bucket['low'],
            'high': (bucket['size'] / 2) + bucket['low'],
            'size': (bucket['size'] / 2) + bucket['low'] - bucket['low'],
            'leftcounter': bucket['leftcounter'],
            'rightcounter': bucket['leftcounter']
        }
        bucket2 = {
            'low': bucket1['low'],
            'high': bucket['high'],
            'size': bucket1['size'],
            'leftcounter': bucket['rightcounter'],
            'rightcounter': bucket['rightcounter']
        }
        buckets = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]['low'] < bucket['low'] or self.buckets[i]['low'] >= bucket['high']:
                buckets.append(self.buckets[i])
            else:
                buckets.append(bucket1)
                buckets.append(bucket2)


    # calculates the error of a single bucket
    def bucketError(self, bucket):
        average = (bucket['leftcounter'] + bucket['rightcounter']) / 2
        lefterror = np.power(bucket['leftcounter'] - average, 2)
        righterror = np.power(bucket['rightcounter'] - average, 2)
        return lefterror + righterror

    # calculates the error of two adjacent buckets
    def adjacentbucketsError(self, bucket1, bucket2):
        return self.bucketError(bucket1) + self.bucketError(bucket2)

    # returns the index of a bucket such that combined  with its successor is smallest among all the pairs.
    def findBestToMerge(self):
        minimum = float('inf')
        index = None
        for i in range(0, self.numbuckets - 1):
            error = self.adjacentbucketsError(self.buckets[i], self.buckets[i + 1])
            if error < minimum:
                minimum = error
                index = i
        return minimum, index
    
    # returns the index of a bucket that has the highest error
    def findBestToSplit(self):
        maximum = float('-inf')
        index = None
        for i in range(0, self.numbuckets):
            error = self.bucketError(self.buckets[i])
            if error > maximum:
                maximum = error
                index = i
        return maximum, index

    def print_buckets(self):
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
