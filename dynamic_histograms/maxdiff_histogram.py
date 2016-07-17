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
import operator

class MaxDiff_Histogram(object):

    def __init__(self, file, numbuckets):
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

    def create_histogram(self, attr, batchsize):
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
                if len(set(sample)) == self.numbuckets * 2:
                    self.compute_histogram(list(set(sample)))
                    break
                    self.print_buckets()
                    self.plot_histogram(attr)
                elif len(set(sample)) > self.numbuckets:
                    self.add_datapoint(float(row[attr_index]))
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.compute_histogram(list(set(sample)))
                        self.print_buckets()
                        self.plot_histogram(attr)

    def compute_histogram(self, sample):
        sample = list(sample)
        sorted_sample = sorted(sample, key=float)
        c = Counter(sorted_sample)
        a = [-1] * (self.numbuckets - 1)
        bucketarea = {}
        for i in range(0, len(sorted_sample) - 2):
            freq = c[sorted_sample[i]]
            spread = sorted_sample[i + 1] - sorted_sample[i]
            area = spread * freq
            area1 = (sorted_sample[i + 2] - sorted_sample[i + 1]) * c[sorted_sample[i + 1]]
            delta = abs(area1 - area)
            if self.checkDifference(delta, a):
                #print "adding " + str(delta)
                a = self.addArea(delta, a, bucketarea, sorted_sample[i])
                #a = stats[0]
                #bucketarea = stats[1]
        print a
        self.arrangeBuckets(c, a, bucketarea, sorted_sample)

    def addArea(self, area, a, bucketarea, value):
        #aprime = []
        length = len(a)
        for i in range(0, len(a)):
            if area == a[i]:
                if bucketarea.has_key(area):
                    bucketarea[area].append(value)
                else:
                    bucketarea[area] = [value]
                #aprime.append(area)
                #aprime.append(a[i])
                a[i:i] = [area]
                area = -2
            elif area > a[i]:
                bucketarea[area] = [value]
                #aprime.append(area)
                a[i:i] = [area]
                area = -2
                if bucketarea.has_key(a[i]):
                    del bucketarea[a[i]]
            #else:  
                #aprime.append(a[i])
            #if len(aprime) == len(a):
            #    break
        #return aprime, bucketarea
        #a = sorted(aprime, key=float)
        # = sorted(aprime, key=float)
        a = sorted(a, reverse=True)[0:length]
        return a

    def checkDifference(self, area, a):
        for i in range(0, len(a)):
            if area >= a[i]:
                return True
        return False

    def arrangeBuckets(self, counter, areas, bucketarea, sample):
        boundaries = sorted(bucketarea.items(), key=operator.itemgetter(1))
        print bucketarea.items()
        print boundaries
        #for i in range(0, len(boundaries)):
        #    self.buckets[i]['low'] = boundaries[i][]

    def add_datapoint(self, value):
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            #if self.buckets[0]['frequency'] >= self.threshold:
                #self.thresholdReached(self.buckets[0], N, sample, attr, l)
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            #if self.buckets[self.numbuckets - 1]['frequency'] >= self.threshold:
                #self.thresholdReached(self.buckets[self.numbuckets - 1], N, sample, attr, l)
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    #if self.buckets[i]['frequency'] >= self.threshold:
                        #self.thresholdReached(self.buckets[i], N, set(sample), attr, l)

    def plot_histogram(self, attr):
        bins = []
        frequency = []
        for bucket in self.buckets:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
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

    def print_buckets(self):
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"