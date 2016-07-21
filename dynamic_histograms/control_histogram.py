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

class Control_Histogram(object):

    """
    This class models an instance of a control histogram, which is equi-width and stretches 
    its bucket boundaries to include values that are beyond the leftmost and rightmost buckets.
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

    def create_histogram(self, attr, batchsize):
        """Reads through the file and creates the histogram, adding in data points as they are being read."""
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
                    self.create_initial_histogram(N, set(sample))
                    self.plot_histogram(attr)
                elif len(set(sample)) > self.numbuckets:
                    self.add_datapoint(float(row[attr_index]))
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_histogram(attr)

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
                elif sorted_sample[j] > self.buckets[i]['high']:
                    break
            low = self.buckets[i]['high']
    
    def add_datapoint(self, value):
        """Increases the count of the bucket the value belongs in the histogram."""
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
        """It plots the histogram. """
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
        plt.title(r'$\mathrm{Control Histogram\ of\ ' + attr + '}$')
        #plt.show()
        path = "control" + str(self.counter) + ".jpg"
        #path = 'control' + str(self.counter)
        #plt.savefig()
        plt.savefig(path)
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
