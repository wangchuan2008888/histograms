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
from scipy.stats import chisquare

# REQUIRES TESTING
class DC_Histogram(object):

    # initializes the class with a default number of four buckets
    def __init__(self, file, numbuckets):
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


    # implements a dynamic compressed histogram while reading the file
    # in the middle of implementing the chisquare part and figuring out how exactly to distinguish 
    # between regular and singleton buckets (I'm thinking of just keeping track of how many unique 
    # values each bucket is representing)
    def create_dc_histogram(self, attr, alpha, batchsize):
         N = 0
         #n = 0
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
                #n = len(set(sample))
                if len(set(sample)) == self.numbuckets:
                    # sample = map(float, sample)
                    sorted_sample = sorted(sample, key=float)
                    buckets = sorted(list(set(sample)), key=float)
                    c = Counter(sorted_sample)
                    # print sorted_sample
                    # print buckets
                    # print c
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
                    #sample.append(float(row[attr_index]))
                    #N = len(set(sample))
                    #N += 1
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_dc_histogram(attr)
                    self.add_datapoint(float(row[attr_index]))
                    chitest = self.chisquaretest(N)
                    if chitest[1] < alpha:
                        print chitest
                        print "number of records read: " + str(N)
                        #self.print_buckets()
                        self.plot_dc_histogram(attr)
                        #self.print_buckets()
                        for i in range(0, self.numbuckets):
                            if len(self.buckets[i]['unique']) == 1 and self.buckets[i]['frequency'] < N / self.numbuckets:
                                self.buckets[i]['regular'] = True
                        self.redistributeRegulars(sample)
                        for i in range(0, self.numbuckets):
                            if self.buckets[i]['regular'] == True:
                                if len(self.buckets[i]['unique']) == 1 and self.buckets[i]['frequency'] > N / self.numbuckets:
                                    self.buckets[i]['regular'] = False
                        #self.print_buckets()
                        self.plot_dc_histogram(attr)

    def redistributeRegulars(self, sample):
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
        #print c
        #print leftoverunique
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
                            #self.buckets[i]['frequency'] += c[unique[j]]
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
                        #leftoverunique = unique
                        width = (max(unique) - min(unique)) / (self.numbuckets - i)
                        self.buckets[i]['high'] = self.buckets[i]['low'] + width
                        self.buckets[i]['size'] = width
                        self.buckets[i]['frequency'] = equalfreq
                        self.buckets[i]['unique'] = []
                        for j in range(0, len(unique)):
                            if unique[j] >= self.buckets[i]['low'] and unique[j] < self.buckets[i]['high']:
                                self.buckets[i]['unique'].append(unique[j])
                                #self.buckets[i]['frequency'] += c[unique[j]]
                        #low = unique[width]
                        #unique = [width:]
                        low = self.buckets[i]['high']
                    #print leftoverunique
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
                    
    # performs the chi-square statistic test
    def chisquaretest(self, N):
        creg = []
        for b in self.buckets:
            #if len(b['unique']) > 1:
            if b['regular'] == True:
            #if b['frequency'] <= N / self.numbuckets:
                creg.append(b['frequency'])
        avg = np.mean(creg)
        cavg = []
        for i in range(0, len(creg)):
            cavg.append(avg)
        return chisquare(creg, f_exp=cavg)
                    
    # this method adds data points to the histograms, adjusting the end bucket partitions if necessary
    def add_datapoint(self, value):
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
            self.buckets[self.numbuckets - 1]['high'] = value
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


    # plots a histogram via matplot.pyplot
    def plot_dc_histogram(self, attr):
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
        high = self.buckets[0]['low']
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
            #assert high == self.buckets[i]['low']
            high = self.buckets[i]['high']
         