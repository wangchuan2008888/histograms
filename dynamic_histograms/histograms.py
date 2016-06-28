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

class Histogram(object):

    # initializes the class with a default number of four buckets
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

    # determines the frequency of 
    def frequency_over_range(attr, sample, low, high):
        frequency = 0
        for val in list(sample[attr]):
            if low <= val and val < high:
                frequency += 1
        return frequency

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
        #buckets = []
        bins = []
        frequency = []
        #size = 0
        for bucket in self.buckets:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
        bins.append(bucket['high'])
        #plt.bar(buckets, frequency, width=size)
        #plt.axis([self.minimum - size, self.maximum + size, 0, frequency[0] + (frequency[0] / 2)])

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
        plt.show()

        # add a 'best fit' line
        # y = mlab.normpdf( bins, mu, sigma)
        #l = plt.plot(bins)

        #plt.xlabel('Smarts')
        #plt.ylabel('Probability')
        #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
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
            frac = float((min(high, self.buckets[i]['high']) - max(low, self.buckets[i]['low']) + 1) / (self.buckets[i]['high'] - self.buckets[i]['low'] + 1))
            #print frac
            self.buckets[i]['frequency'] = max(self.buckets[i]['frequency'] + (alpha * esterr * frac * (self.buckets[i]['frequency'] / est)), 0)
    
    # the algorithm for restructing histograms 
    # m is a parameter that we call the merge threshold. In most of the experiments, m <= 1% was a suitable choice
    # s is a parameter that we call the split threshold. In the experiments, we used s=10% 


    # MAYBE I FIXED EVERYTHING???? INTENSTIVE TESTING REQUIRED

    def restructureHist(self, m, s):
        #min = float('inf')
        #index = 0
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
                #localmin = abs(self.buckets[i + 1]['frequency'] - self.buckets[i]['frequency'])

                #if localmax < min:
                #    min = localmax
                #    index = i
            mintuple = min(maxfreq, key=itemgetter(0))
            if mintuple[0] <= m * self.maxlength:
                print "length: " + str(len(bucketruns))
                #print "index: " + str(min)
                #mergebuckets(self.buckets[index], self.buckets[index + 1])
                bucketruns = self.mergeruns(bucketruns, mintuple[1], mintuple[2])
                freebuckets += 1
                #index = 0
            else:
                break
        print "done merging (ish)\n"
        
        k = round(s * self.numbuckets)

        #while len(highbuckets) < k:
        unmergedbuckets = []
        for b in self.buckets:
            if b['merge'] == False:
                unmergedbuckets.append(b)
        frequencies = [b['frequency'] for b in unmergedbuckets]
        highfrequencies = nlargest(k, frequencies)
        totalfreq = sum(highfrequencies)
        #for i in highbuckets:
        #    totalfreq += i
        highbuckets = []
        for b in self.buckets:
            if b['frequency'] in highfrequencies:
                highbuckets.append(b)

        # merging each run that has more than one bucket in it, meaning those buckets should be merged together
        for l in bucketruns:
            if len(l) != 1:
                for i in range(0, len(l) - 1):
                    self.mergebuckets(l[i], l[i + 1])

        for b in highbuckets:
            #numsplit = round((b['frequency'] / totalfreq) * freebuckets)
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
            elif bucket['low'] == b['low'] and bucket['high'] == b['high'] and bucket['frequency'] != b['frequency']:
                low = b['low']
                high = low + size
                for i in range(0, numsplit):
                    newb = {
                        'low': low,
                        'high': high,
                        'frequency': round(b['frequency'] / numsplit),
                        'size': high - low,
                        'merge': False
                    } 
                    low = high
                    if (i == numsplit - 2):
                        high = b['high'] - size
                    else:
                        high = low + size
                    newbuckets.append(newb)
        self.buckets = newbuckets


    # buckets, b1, and b2 are all lists of buckets

    def mergeruns(self, buckets, b1, b2):
        print "### BUCKET 1 ###"
        print b1
        print "### END ###\n"
        print "### BUCKET 2 ###"
        print b2
        print "### END ###\n"
        for b in b1:
            b['merge'] = True
        for b in b2:
            b['merge'] = True
        merged = b1 + b2
        print "### MERGED ###"
        print merged
        print "###END\n"
        newbuckets = []
        print "### BEFORE ###"
        print buckets
        print "### END ###"
        prev = len(buckets)
        print "previous length: " + str(prev)
        for b in buckets:
            #if set(b) == set(b2):
            if self.checkBucketLists(b, b2) == True:
                pass
            elif self.checkBucketLists(b, b1) == False:
            #set(b) != set(b1):
                newbuckets.append(b)
            elif self.checkBucketLists(b, b1) == True: 
            #set(b) == set(b1):
                newbuckets.append(merged)
        print "### AFTER ###"
        print newbuckets
        print "### END ###"
        new = len(newbuckets)
        print "new length: " + str(new)
        assert new < prev
        #raise KeyError
        return newbuckets

    # checks if two lists of buckets are the same
    def checkBucketLists(self, b1, b2):
        for x in b1:
            for y in b2:
                if self.equalBuckets(x, y) == False:
                    return False
        return True


    # this method checks if two buckets (which are dicts) are the same
    def equalBuckets(self, b1, b2):
        if b1['low'] != b2['low'] or b1['high'] != b2['high'] or b1['frequency'] != b2['frequency'] or b1['merge'] != b2['merge'] or b1['size'] != b2['size']:
            return False
        else:
            return True



    # merging b1 with b2, resulting bucket has boundaries of b1.low and b2.high
    def mergebuckets(self, b1, b2):
        buckets = []
        for bucket in self.buckets:
            if bucket['low'] == b2['low'] and bucket['high'] == b2['high']:
                pass
            elif bucket['low'] != b1['low'] and bucket['high'] != b1['high']:
                buckets.append(bucket)
            elif bucket['low'] == b1['low'] and bucket['high'] == b1['high']:
                newbucket = {
                    'low': b1['low'],
                    'high': b2['high'],
                    'frequency': b1['frequency'] + b2['frequency'],
                    'size': b2['high'] - b1['low'],
                    'merge': False
                }
                buckets.append(newbucket)
        self.buckets = buckets
            
