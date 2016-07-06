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
from heapq import nlargest
from operator import itemgetter
import csv
from collections import Counter
from scipy.stats import chisquare

class SF_Histogram(object):

    # initializes the class with a default number of 10 buckets
    def __init__(self, frame, min, max):
        self.frame = frame
        self.maxlength = len(self.frame.index)
        self.numbuckets = 10
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0, 
                'frequency': 0,
                'size': 0,
                'merge': False
            })
        self.buckets = buckets
        self.min = min
        self.max = max

    # creates the initial histogram from the sample on the atttribute, using only the sample's min and max
    # since the intial self-tuning histogram does not look at the data and assumes a frequency of maximum 
    # observations / # of buckets for each bucket
    def create_initial_sf_histogram(self, attr):
        range = math.ceil(self.max - self.min) # want to make sure we capture the maximum element in the last bucket
        size = math.ceil(range / self.numbuckets)
        low = self.min
        high = self.min + size
        for bucket in self.buckets:
            bucket['low'] = low
            bucket['high'] = high
            bucket['frequency'] = round(self.maxlength / self.numbuckets)
            bucket['size'] = size
            low = high
            high += size

    # plots a histogram via matplot.pyplot. this is the intial histogram of the self-tuning histogram which is both equi-depth
    # and equi-width (because the intial histogram does not look at the data frequencies)
    def plot_sf_histogram(self, attr):
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
        axes.set_xlim([0, self.max + widths[0]])
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Histogram\ of\ ' + attr + '}$')
        plt.show()

        # add a 'best fit' line
        # y = mlab.normpdf( bins, mu, sigma)
        #l = plt.plot(bins)
        #plt.axis([40, 160, 0, 0.03])

    '''
    UpdateFreq
    Inputs: h, rangelow, rangehigh, act
    Outputs: h with updated bucket frequencies
    '''
    # alpha is a dampening factor in the range 0.5 to 1 to make sure that bucket frequencies are not
    # modified too much in response to errors, as this may lead to oversensitive
    # or unstable histograms.

    def updateFreq(self, low, high, act, alpha):
        b = []
        for i in range(0, self.numbuckets):
            if (self.buckets[i]['low'] < low and self.buckets[i]['high'] > low):
                b.append(i)
            elif (self.buckets[i]['low'] >= low and self.buckets[i]['high'] <= high):
                b.append(i)
            elif (self.buckets[i]['low'] < high and self.buckets[i]['high'] > high):
                b.append(i)
        est = 0
        for i in b:
            est += self.buckets[i]['frequency']
        esterr = act - est
        for i in b:
            frac = (min(high, self.buckets[i]['high']) - max(low, self.buckets[i]['low']) + 1) / (self.buckets[i]['size'] + 1)
            self.buckets[i]['frequency'] = max(self.buckets[i]['frequency'] + (alpha * esterr * frac * (self.buckets[i]['frequency'] / est)), 0)
    
    # the algorithm for restructing histograms 
    # m is a parameter that we call the merge threshold. In most of the experiments, m <= 1% was a suitable choice
    # s is a parameter that we call the split threshold. In the experiments, we used s=10% 

    # all seems okay, TESTING IS REQUIRED   

    def restructureHist(self, m, s):
        freebuckets = 0
        bucketruns = []
        for b in self.buckets:
            bucketruns.append([b])
        while True:
            maxfreq = []
            for i in range(0, len(bucketruns) - 1):
                localmax = float('-inf')
                tuple = []
                for b1 in bucketruns[i]:
                    for b2 in bucketruns[i + 1]:
                        diff = abs(b2['frequency'] - b1['frequency'])
                        if diff > localmax:
                            localmax = diff
                            tuple = [localmax, bucketruns[i], bucketruns[i + 1]]
                maxfreq.append(tuple)
            mintuple = min(maxfreq, key=itemgetter(0))
            if mintuple[0] <= m * self.maxlength:
                bucketruns = self.mergeruns(bucketruns, mintuple[1], mintuple[2])
                freebuckets += 1
            else:
                break
        
        k = round(s * self.numbuckets)

        unmergedbuckets = []
        for b in self.buckets:
            if b['merge'] == False:
                unmergedbuckets.append(b)
        frequencies = [b['frequency'] for b in unmergedbuckets]
        highfrequencies = nlargest(k, frequencies)
        totalfreq = sum(highfrequencies)
        highbuckets = []
        for b in self.buckets:
            if b['frequency'] in highfrequencies:
                highbuckets.append(b)

        # merging each run that has more than one bucket in it, meaning those buckets should be merged together
        for l in bucketruns:
            if len(l) != 1:
                self.mergebuckets(l)

        for b in highbuckets:
            self.splitbucket(b, freebuckets, totalfreq)
        
        self.numbuckets = len(self.buckets)

    # splits the bucket into the appropriate number and inserts that into the buckets list kept with the histogram
    # numfree - # of free buckets
    # totalfreq - total frequency of the buckets that need to be split

    def splitbucket(self, b, numfree, totalfreq):
        numsplit = round(((b['frequency'] / totalfreq) * numfree) + 1)
        size = b['size'] / numsplit
        newbuckets = []
        for bucket in self.buckets:
            if bucket['low'] != b['low'] and bucket['high'] != b['high'] and bucket['frequency'] != b['frequency']:
                newbuckets.append(bucket)
            elif bucket['low'] == b['low'] and bucket['high'] == b['high'] and bucket['frequency'] == b['frequency']:
                low = b['low']
                high = low + size
                for i in range(0, int(numsplit)):
                    newb = {
                        'low': low,
                        'high': high,
                        'frequency': b['frequency'] / numsplit,
                        'size': high - low,
                        'merge': False
                    } 
                    low = high
                    if (i == numsplit - 2):
                        high = b['high']
                    else:
                        high = low + size
                    newbuckets.append(newb)
        self.buckets = newbuckets
        self.numbuckets = len(newbuckets)


    # buckets, b1, and b2 are all lists of buckets

    def mergeruns(self, buckets, b1, b2):
        for b in b1:
            b['merge'] = True
        for b in b2:
            b['merge'] = True
        merged = b1 + b2
        newbuckets = []
        prev = len(buckets)
        for b in buckets:
            if self.checkBucketLists(b, b2) == True:
                pass
            elif self.checkBucketLists(b, b1) == False:
                newbuckets.append(b)
            elif self.checkBucketLists(b, b1) == True: 
                newbuckets.append(merged)
        new = len(newbuckets)
        assert new < prev
        return newbuckets

    # checks if two lists of buckets are the same
    # assuming that the lsits of buckets are in order if they are the same, i.e. b1[0] = b2[0] and so forth 
    # if they are the same
    def checkBucketLists(self, b1, b2):
        b1length = len(b1)
        b2length = len(b2)
        if b1length != b2length:
            return False
        else:
            for i in range(0, b1length):
                if self.equalBuckets(b1[i], b2[i]) == False:
                    return False
        return True


    # this method checks if two buckets (which are dicts) are the same
    def equalBuckets(self, b1, b2):
        if b1['low'] != b2['low'] or b1['high'] != b2['high'] or b1['frequency'] != b2['frequency'] or b1['merge'] != b2['merge'] or b1['size'] != b2['size']:
            return False
        else:
            return True

    # merging all the buckets in bucketrun into one bucket and inserting that bucket where all the previous
    # unmerged buckets were
    def mergebuckets(self, bucketrun):
        buckets = []
        totalfreq = 0
        low = bucketrun[0]['low']
        for b in bucketrun:
            totalfreq += b['frequency']
        high = b['high']
        for bucket in self.buckets:
            if bucket['low'] < low:
                buckets.append(bucket)
            elif bucket['low'] == bucketrun[0]['low'] and bucket['high'] == bucketrun[0]['high']:
                totalfreq = 0
                low = bucketrun[0]['low']
                for b in bucketrun:
                    totalfreq += b['frequency']
                high = b['high']
                newbucket = {
                    'low': low,
                    'high': high,
                    'frequency': totalfreq,
                    'size': high - low,
                    'merge': False
                }
                buckets.append(newbucket)
            elif bucket['low'] > low and bucket['low'] < high:
                pass
            else:
                buckets.append(bucket)
        self.buckets = buckets
        self.numbuckets = len(buckets)
            

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
                'merge': False,
                'unique': [],
                'regular': False
            })
        self.buckets = buckets


    # implements a dynamic compressed histogram while reading the file
    # in the middle of implementing the chisquare part and figuring out how exactly to distinguish 
    # between regular and singleton buckets (I'm thinking of just keeping track of how many unique 
    # values each bucket is representing)
    def create_dc_histogram(self, attr, alpha):
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
                    self.add_datapoint(float(row[attr_index]))
                    chitest = self.chisquaretest(N)
                    if chitest[1] < alpha:
                        print chitest
                        print "number of records read: " + str(N)
                        for i in range(0, self.numbuckets):
                            if len(self.buckets[i]['unique']) == 1 and self.buckets[i]['frequency'] < N / self.numbuckets:
                                self.buckets[i]['regular'] = True
                        self.redistributeRegular()
                        for i in range(0, self.numbuckets):
                            if self.buckets[i]['regular'] == True:
                                if len(self.buckets[i]['unique']) == 1 and self.buckets[i]['frequency'] > N / self.numbuckets:
                                    self.buckets[i]['regular'] = False
                        print self.buckets
                        self.plot_dc_histogram(attr)


    # redistributes regular buckets to equalize their counts      
    def redistributeRegular(self):
        N = 0
        beta = 0
        leftoverunique = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]['regular'] == True:
                beta += 1
                N += self.buckets[i]['frequency']
                leftoverunique = list(set(leftoverunique) | set(self.buckets[i]['unique']))
        equalfreq = N / beta
        leftover = 0
        low = self.buckets[0]['low']
        for i in range(0, self.numbuckets):
            if self.buckets[i]['regular'] == True:
                perc = equalfreq / self.buckets[i]['frequency']
                if perc < 1:
                    # we need to shrink bucket
                    self.buckets[i]['low'] = low
                    leftover += self.buckets[i]['size'] - (self.buckets[i]['size'] * perc)
                    self.buckets[i]['size'] *= perc
                    self.buckets[i]['high'] = self.buckets[i]['low'] + self.buckets[i]['size']
                    self.buckets[i]['frequency'] = equalfreq
                    low = self.buckets[i]['high']
                    unique = []
                    for j in range(0, len(leftoverunique)):
                        if leftoverunique[j] >= self.buckets[i]['low'] and leftoverunique[j] < self.buckets[i]['high']:
                            unique.append(leftoverunique[j])
                    self.buckets[i]['unique'] = unique
                elif perc > 1:
                    # we need to expand bucket
                    self.buckets[i]['low'] = low
                    leftover -= (self.buckets[i]['size'] * perc) - self.buckets[i]['size']
                    self.buckets[i]['size'] *= perc
                    self.buckets[i]['high'] = self.buckets[i]['low'] + self.buckets[i]['size']
                    self.buckets[i]['frequency'] = equalfreq
                    low = self.buckets[i]['high']
                    unique = []
                    for j in range(0, len(leftoverunique)):
                        if leftoverunique[j] >= self.buckets[i]['low'] and leftoverunique[j] < self.buckets[i]['high']:
                            unique.append(leftoverunique[j])
                    self.buckets[i]['unique'] = unique
                    leftoverunique = list(set(leftoverunique) - set(self.buckets[i]['unique']))
                else:
                    self.buckets[i]['low'] = low
                    self.buckets[i]['high'] = self.buckets[i]['size'] + self.buckets[i]['low']
                    low = self.buckets[i]['high']
            else:
                low = self.buckets[i]['high']
                    
    # performs the chi-square statistic test
    def chisquaretest(self, N):
        creg = []
        for b in self.buckets:
            if len(b['unique']) > 1:
            #if b['regular'] == True:
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
            if self.buckets[0]['regular'] == False and len(self.buckets[0]['unique']) > 1:
                self.buckets[0]['regular'] = True
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            if value not in self.buckets[self.numbuckets - 1]['unique']:
                self.buckets[self.numbuckets - 1]['unique'].append(value)
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
        axes.set_xlim([0, self.buckets[self.numbuckets - 1]['high'] * 1.5])
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Histogram\ of\ ' + attr + '}$')
        plt.show()
         