"""
It constructs a dynamic compressed histogram from the sample given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
from collections import Counter
from scipy.stats import chisquare

class DC_Histogram(object):

    """
    This class models an instance of a dynamically generated compressed histogram, which has at least one equi-depth
    bucket, with the other buckets being singleton buckets. 
    """

    def __init__(self, file, numbuckets):

        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """

        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0, 
                'frequency': 0,
                'size': 0,
                'unique': [],
                'regular': False
            })
        self.buckets = buckets

    def create_dc_histogram(self, attr, alpha, batchsize):
        """Reads in data from the file, extending the buckets of the histogram is the values are beyond 
        it, and checks to see if the probability that the counts in the equi-depth buckets are not uniformly 
        distributed is statistically significant (less than alpha) and if so, redistributes the regular buckets."""
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
                        self.buckets[i]['frequency'] = c[buckets[i]]
                        self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                        if buckets[i] not in self.buckets[i]['unique']:
                            self.buckets[i]['unique'].append(buckets[i])
                        if self.buckets[i]['regular'] == False and len(self.buckets[i]['unique']) > 1:
                            buckets[i]['regular'] = True
                elif len(set(sample)) > self.numbuckets:
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_dc_histogram(attr)
                    self.add_datapoint(float(row[attr_index]))
                    chitest = self.chisquaretest(N)
                    if chitest[1] < alpha:
                        print chitest
                        print "number of records read: " + str(N)
                        self.plot_dc_histogram(attr)
                        for i in range(0, self.numbuckets):
                            if len(self.buckets[i]['unique']) == 1 and self.buckets[i]['frequency'] < N / self.numbuckets:
                                self.buckets[i]['regular'] = True
                        self.redistributeRegulars(sample)
                        for i in range(0, self.numbuckets):
                            if self.buckets[i]['regular'] == True:
                                if len(self.buckets[i]['unique']) == 1 and self.buckets[i]['frequency'] > N / self.numbuckets:
                                    self.buckets[i]['regular'] = False
                        self.plot_dc_histogram(attr)

    def redistributeRegulars(self, sample):
        """This method redistributes the regular buckets."""
        N = 0
        beta = 0
        leftoverunique = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]['regular'] == True:
                beta += 1
                N += self.buckets[i]['frequency']
                leftoverunique = list(set(leftoverunique) | set(self.buckets[i]['unique']))
        leftoverunique = sorted(leftoverunique, key=float)
        equalfreq = N / beta
        print N, beta
        leftover = 0
        low = self.buckets[0]['low']
        sorted_sample = sorted(sample, key=float)
        c = Counter(sorted_sample)
        width = 0
        for i in range(0, self.numbuckets):
            if self.buckets[i]['regular'] == True:
                if len(leftoverunique) == 0:
                    self.buckets[i]['low'] = low
                    self.buckets[i]['high'] = self.buckets[i]['low'] + width
                    self.buckets[i]['size'] = width
                    self.buckets[i]['frequency'] = equalfreq
                    self.buckets[i]['unique'] = []
                    for j in range(0, len(unique)):
                        if unique[j] >= self.buckets[i]['low'] and unique[j] < self.buckets[i]['high']:
                            self.buckets[i]['unique'].append(unique[j])
                    low = self.buckets[i]['high']
                else:
                    unique = []
                    self.buckets[i]['low'] = low
                    self.buckets[i]['frequency'] = 0
                    for j in range(0, len(leftoverunique)):
                        self.buckets[i]['frequency'] += c[leftoverunique[j]]
                        unique.append(leftoverunique[j])
                        if self.buckets[i]['frequency'] >= equalfreq:
                            self.buckets[i]['frequency'] = equalfreq
                            if j == len(leftoverunique) - 1:
                                self.buckets[i]['high'] = leftoverunique[j]
                            else:
                                self.buckets[i]['high'] = leftoverunique[j + 1]
                            break
                    if j == len(leftoverunique) - 1:
                        self.buckets[i]['high'] = leftoverunique[j]
                    self.buckets[i]['unique'] = unique
                    self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                    leftoverunique = list(set(leftoverunique) - set(unique))
                    leftoverunique = sorted(leftoverunique, key=float)
                    low = self.buckets[i]['high']
                    if len(leftoverunique) == 0:
                        width = (max(unique) - min(unique)) / (self.numbuckets - i)
                        self.buckets[i]['high'] = self.buckets[i]['low'] + width
                        self.buckets[i]['size'] = width
                        self.buckets[i]['frequency'] = equalfreq
                        self.buckets[i]['unique'] = []
                        for j in range(0, len(unique)):
                            if unique[j] >= self.buckets[i]['low'] and unique[j] < self.buckets[i]['high']:
                                self.buckets[i]['unique'].append(unique[j])
                        low = self.buckets[i]['high']
            else:
                self.buckets[i]['low'] = unique[0]
                self.buckets[i]['high'] = leftoverunique[0]
                self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']

    # redistributes regular buckets to equalize their counts
    # THIS ENTIRE METHOD IS WRONG JUST DO IT SIMPLY FOR NOW AND THEN GET BETTER ON IT ONCE YOU KNOW IT WORKS  
    # def redistributeRegular(self):
    #     N = 0
    #     beta = 0
    #     leftoverunique = []
    #     for i in range(0, self.numbuckets):
    #         if self.buckets[i]['regular'] == True:
    #             beta += 1
    #             N += self.buckets[i]['frequency']
    #             leftoverunique = list(set(leftoverunique) | set(self.buckets[i]['unique']))
    #     equalfreq = N / beta
    #     print N, beta
    #     leftover = 0
    #     low = self.buckets[0]['low']
    #     for i in range(0, self.numbuckets):
    #         if self.buckets[i]['regular'] == True:
    #             perc = equalfreq / self.buckets[i]['frequency']
    #             self.buckets[i]['low'] = low
    #             self.buckets[i]['size'] *= perc
    #             self.buckets[i]['high'] = self.buckets[i]['size'] + self.buckets[i]['low']
    #             #if i != self.numbuckets - 1 and self.buckets[i + 1]['regular'] == False and self.buckets[i]['high'] > self.buckets[i + 1]['low']:
    #             #    self.buckets[i]['high'] = self.buckets[i + 1]['low']
    #             #    self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
    #             self.buckets[i]['frequency'] = equalfreq
    #             low = self.buckets[i]['high']
    #             #print perc
    #             if perc < 1:
    #                 # we need to shrink bucket
    #                 #leftover += self.buckets[i]['size'] - (self.buckets[i]['size'] * perc)
    #                 unique = []
    #                 for j in range(0, len(leftoverunique)):
    #                     if leftoverunique[j] >= self.buckets[i]['low'] and leftoverunique[j] < self.buckets[i]['high']:
    #                         unique.append(leftoverunique[j])
    #                 self.buckets[i]['unique'] = unique
    #                 leftoverunique = list(set(leftoverunique) - set(self.buckets[i]['unique']))
    #             elif perc > 1:
    #                 # we need to expand bucket
    #                 #leftover -= (self.buckets[i]['size'] * perc) - self.buckets[i]['size']
    #                 unique = []
    #                 for j in range(0, len(leftoverunique)):
    #                     if leftoverunique[j] >= self.buckets[i]['low'] and leftoverunique[j] < self.buckets[i]['high']:
    #                         unique.append(leftoverunique[j])
    #                 self.buckets[i]['unique'] = unique
    #                 leftoverunique = list(set(leftoverunique) - set(self.buckets[i]['unique']))
    #         else:
    #             #self.buckets[i]['low'] = self.buckets[i - 1]['high']
    #             low = self.buckets[i]['high']
    #             #self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                    
    def chisquaretest(self, N):
        """Performs the chi-square statistic test and returns the p-value and the chi-squared value."""
        creg = []
        for b in self.buckets:
            if b['regular'] == True:
                creg.append(b['frequency'])
        avg = np.mean(creg)
        cavg = []
        for i in range(0, len(creg)):
            cavg.append(avg)
        return chisquare(creg, f_exp=cavg)
                    
    def add_datapoint(self, value):
        """this method adds data points to the histograms, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            if value not in self.buckets[0]['unique']:
                self.buckets[0]['unique'].append(value)
            if len(self.buckets[0]['unique']) == 1:
                self.buckets[0]['regular'] = False
            if self.buckets[0]['regular'] == False and len(self.buckets[0]['unique']) > 1:
                self.buckets[0]['regular'] = True
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            if value not in self.buckets[self.numbuckets - 1]['unique']:
                self.buckets[self.numbuckets - 1]['unique'].append(value)
            if len(self.buckets[self.numbuckets - 1]['unique']) == 1:
                self.buckets[self.numbuckets - 1]['regular'] = False
            if self.buckets[self.numbuckets - 1]['regular'] == False and len(self.buckets[self.numbuckets - 1]['unique']) > 1:
                self.buckets[self.numbuckets - 1]['regular'] = True
        else:
            for i in range(0, self.numbuckets):
                if value == self.buckets[i]['low']:
                    if value not in self.buckets[i]['unique']:
                        self.buckets[i]['unique'].append(value)
                    self.buckets[i]['frequency'] += 1
                    if self.buckets[i]['regular'] == False and len(self.buckets[i]['unique']) > 1:
                       self.buckets[i]['regular'] = True
                elif value > self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    if value not in self.buckets[i]['unique']:
                        self.buckets[i]['unique'].append(value)
                    if self.buckets[i]['regular'] == False and len(self.buckets[i]['unique']) > 1:
                        self.buckets[i]['regular'] = True
                if len(self.buckets[i]['unique']) == 1:
                    self.buckets[i]['regular'] = False


    def plot_dc_histogram(self, attr):
        """Plots the histogram."""
        bins = []
        frequency = []
        for bucket in self.buckets:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
        bins.append(bucket['high'])

        frequency = np.array(frequency)
        bins = np.array(bins)

        widths = bins[1:] - bins[:-1]

        plt.bar(bins[:-1], frequency, width=widths)

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([self.buckets[0]['low'], self.buckets[self.numbuckets - 1]['high'] * 1.5])
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Histogram\ of\ ' + attr + '}$')
        plt.show()

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        high = self.buckets[0]['low']
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
            high = self.buckets[i]['high']
         