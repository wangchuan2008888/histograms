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
from collections import defaultdict
import operator
import itertools

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
        self.arrangeBuckets(c, a, bucketarea, sorted_sample)

    def addArea(self, area, a, bucketarea, index):
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
        for i in range(0, len(a)):
            if area >= a[i]:
                return True
        return False

    def arrangeBuckets(self, counter, areas, bucketarea, sample):
        boundaries = sorted(bucketarea.items(), key=operator.itemgetter(1))
        low = sample[0]
        values = bucketarea.values()
        values = list(itertools.chain(*values))
        values = sorted(values)
        for i in range(0, len(values)):
            self.buckets[i]['low'] = low
            highindex = values[i]
            self.buckets[i]['high'] = sample[highindex]
            self.buckets[i]['size'] = sample[highindex] - low
            self.buckets[i]['frequency'] = counter[low]
            low = self.buckets[i]['high']
        self.buckets[self.numbuckets - 1]['low'] = self.buckets[self.numbuckets - 2]['high']
        self.buckets[self.numbuckets - 1]['frequency'] = counter[self.buckets[self.numbuckets - 1]['low']]
        self.buckets[i]['high'] = sample[len(sample) - 1] + 1
        self.buckets[i]['size'] = sample[len(sample) - 1] + 1 - self.buckets[self.numbuckets - 1]['low']
        index = 0
        bucket = self.buckets[index]
        for i in range(0, len(sample)):
            if sample[i] >= bucket['high']:
                index += 1
                bucket = self.buckets[index]
            if sample[i] >= bucket['low'] and sample[i] < bucket['high']:
                bucket['frequency'] += counter[sample[i]]


    def add_datapoint(self, value):
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value
            self.buckets[self.numbuckets - 1]['frequency'] += 1
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1

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