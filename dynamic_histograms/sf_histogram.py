'''
Given samples, it constructs the appropriate histogram from the sample

Steffani Gomez(smg1)
'''

from __future__ import division
import numpy as np
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from heapq import nlargest
from operator import itemgetter

class SF_Histogram(object):

    # initializes the class with a default number of 10 buckets
    # def __init__(self, frame, min, max):
    #     self.frame = frame
    #     self.maxlength = len(self.frame.index)
    #     self.numbuckets = 10
    #     buckets = []
    #     for i in range(0, self.numbuckets):
    #         buckets.append({
    #             'low': 0,
    #             'high': 0, 
    #             'frequency': 0,
    #             'size': 0,
    #             'merge': False
    #         })
    #     self.buckets = buckets
    #     self.min = float("inf")
    #     self.max = float("-inf")

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
                'frequency': 0,
                'merge': False
            })
        self.min = float("inf")
        self.max = float("-inf")
        self.buckets = buckets

    # creates the initial histogram from the sample on the atttribute, using only the sample's min and max
    # since the intial self-tuning histogram does not look at the data and assumes a frequency of maximum 
    # observations / # of buckets for each bucket
    def create_initial_histogram(self, s):
        range = math.ceil(self.max - self.min) # want to make sure we capture the maximum element in the last bucket
        size = math.ceil(range / self.numbuckets)
        low = self.min
        high = self.min + size
        for bucket in self.buckets:
            bucket['low'] = low
            bucket['high'] = high
            bucket['frequency'] = round(s / self.numbuckets)
            bucket['size'] = size
            low = high
            high += size

    def create_histogram(self, attr, alpha, m, s, batchsize):
        """l is a tunable parameter (> -1) which influences the upper thresholder of bucket count for all buckets. The appropriate bucket counter is 
        incremented for every record read in. If a bucket counter reaches threshold, the bucket boundaries are recalculated and the threshold is updated."""
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
                if float(row[attr_index]) < self.min:
                    self.min = float(row[attr_index])
                if float(row[attr_index]) > self.max:
                    self.max = float(row[attr_index])
                N += 1
                if len(set(sample)) == self.numbuckets:
                    self.create_initial_histogram(len(set(sample)))
                    self.plot_histogram(attr)
                elif len(set(sample)) > self.numbuckets:
                    #print "number read in: " + str(N)
                    self.add_datapoint(float(row[attr_index]), sample, alpha)
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.print_buckets()
                        print "BEFORE"
                        self.restructureHist(m, s, len(set(sample)))
                        self.print_buckets()
                        self.plot_histogram(attr)

    def sample_on_range(self, sample, rangelow, rangehigh):
        sample = sorted(sample, key=float)
        #print rangelow
        #print rangehigh
        #print sample
        s = []
        for i in range(0, len(sample)):
            #print i
            if sample[i] >= rangelow and sample[i] < rangehigh:
                sample.append(sample[i])
            if sample[i] >= rangehigh:
                break
        return s

    def add_datapoint(self, value, sample, alpha):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
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


    # def add_datapoint(self, value, sample, alpha):
    #     """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
    #     if value < self.buckets[0]['low']:
    #         self.buckets[0]['low'] = value
    #         #self.buckets[0]['frequency'] += 1
    #         high = self.buckets[0]['high']
    #         self.buckets[0]['size'] = high - value
    #         s = self.sample_on_range(sample, rangelow=value, rangehigh=high)
    #         size = len(s)
    #         self.updateFreq(value, high, size, alpha)
    #     elif value > self.buckets[self.numbuckets - 1]['high']:
    #         self.buckets[self.numbuckets - 1]['high'] = value
    #         #self.buckets[self.numbuckets - 1]['frequency'] += 1
    #         low = self.buckets[self.numbuckets - 1]['low']
    #         self.buckets[self.numbuckets - 1]['size'] = value - low
    #         s = self.sample_on_range(sample, rangelow=low, rangehigh=value)
    #         size = len(s)
    #         self.updateFreq(low, value, size, alpha)
    #     else:
    #         for i in range(0, self.numbuckets):
    #             if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
    #                 #self.buckets[i]['frequency'] += 1
    #                 low = self.buckets[i]['low']
    #                 high = self.buckets[i]['high']
    #                 s = self.sample_on_range(sample, rangelow=low, rangehigh=high)
    #                 size = len(s)
    #                 self.updateFreq(low, high, size, alpha)

    # plots a histogram via matplot.pyplot. this is the intial histogram of the self-tuning histogram which is both equi-depth
    # and equi-width (because the intial histogram does not look at the data frequencies)
    def plot_histogram(self, attr):
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

        plt.bar(bins[:-1], frequency, width=widths, edgecolor=['black'])

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([self.buckets[0]['low'] - self.buckets[0]['size'], self.buckets[self.numbuckets - 1]['high'] * 1.5])
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Histogram\ of\ ' + attr + '}$')
        plt.show()


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

    def restructureHist(self, m, s, size):
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
            if len(maxfreq) > 0:
                mintuple = min(maxfreq, key=itemgetter(0))
            else:
                break
            if mintuple[0] <= m * size:
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

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""        
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
            