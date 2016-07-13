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

class Equidepth_Histogram(object):

    def __init__(self, f, numbuckets):
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

    def create_histogram(self, attr, l, batchsize):
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
                    self.create_initial_histogram(N, set(sample), l)
                    #self.print_buckets()
                    self.plot_histogram(attr)
                elif len(set(sample)) > self.numbuckets:
                    self.add_datapoint(float(row[attr_index]), N, sample, attr, l)
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        #self.print_buckets()
                        self.plot_histogram(attr)

    def create_initial_histogram(self, N, sample, l):
        sample = list(sample)
        sorted_sample = sorted(sample, key=float)
        for i in range(0, self.numbuckets):
            self.buckets[i]['low'] = sorted_sample[i]
            if i == self.numbuckets - 1:
                self.buckets[i]['high'] = sorted_sample[i] + 1
            else:
                self.buckets[i]['high'] = sorted_sample[i + 1]
            self.buckets[i]['frequency'] = N / self.numbuckets
            self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
        self.threshold = (2 + l) * (N / self.numbuckets)

    # since the pseudocode doesn't mention what to do about values that fall outside of buckets, I extend those bucket
    # boundaries to include values that are not included in the boundary range 
    def add_datapoint(self, value, N, sample, attr, l):
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            if self.buckets[0]['frequency'] >= self.threshold:
                self.thresholdReached(self.buckets[0], N, sample, attr, l)
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            if self.buckets[self.numbuckets - 1]['frequency'] >= self.threshold:
                self.thresholdReached(self.buckets[self.numbuckets - 1], N, sample, attr, l)
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    if self.buckets[i]['frequency'] >= self.threshold:
                        self.thresholdReached(self.buckets[i], N, set(sample), attr, l)

    def thresholdReached(self, bucket, N, sample, attr, l):
        for i in range(0, self.numbuckets - 1):
            if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.threshold:
                self.mergebuckets(self.buckets[i], self.buckets[i + 1])
                self.splitbucket(bucket)
            else:
                self.computehistogram(sample, N)
                self.threshold = (2 + l) * (N / self.numbuckets)
        print "RESTRUCTURING number read in: " + str(N)
        #self.print_buckets()
        self.plot_histogram(attr)

    def computehistogram(self, sample, N):
        sample = list(sample)
        sample = sorted(sample, key=float)
        for i in range(0, self.numbuckets):
            self.buckets[i]['low'] = sample[int(round(i * (len(sample) / self.numbuckets)))]
            if i == self.numbuckets - 1:
                self.buckets[i]['high'] = sample[int(round(i * (len(sample) / self.numbuckets)))] + 1
            else:
                self.buckets[i]['high'] = sample[int(round(i + 1 * (len(sample) / self.numbuckets)))]
            self.buckets[i]['frequency'] = N / self.numbuckets

    def mergebuckets(self, bucket1, bucket2):
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
            elif self.buckets[i]['low'] == bucket2['low'] and self.buckets[i]['high'] == bucket2['high']:
                pass
            else:
                buckets.append(self.buckets[i])
        self.buckets = buckets

    def splitbucket(self, bucket):
        bucket1 = {
            'low': bucket['low'],
            'high': bucket['size'] / 2 + bucket['low'],
            'size': (bucket['size'] / 2 + bucket['low']) - bucket['low'],
            'frequency': self.threshold / 2
        }
        bucket2 = {
            'low': bucket1['high'], 
            'high': bucket['high'],
            'size': bucket['high'] - bucket1['high'],
            'frequency': self.threshold / 2
        }
        buckets = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]['low'] == bucket['low'] and self.buckets[i]['high'] == bucket['high']:
                buckets.append(bucket1)
                buckets.append(bucket2)
            else:
                buckets.append(self.buckets[i])
        self.buckets = buckets

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