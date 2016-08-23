"""
It constructs a dynamic compressed histogram from the sample given.

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
import random
import user_distribution
import json
from scipy import stats

upper_factor = 3

class DC_Histogram(object):

    """
    This class models an instance of a dynamically generated compressed histogram, which has at least one equi-depth
    bucket, with the other buckets being singleton buckets. 
    """

    def __init__(self, file, numbuckets, outputpath):

        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """
        self.outputpath = outputpath
        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0, 
                'frequency': 0,
                'size': 0,
                'regular': True
            })
        self.buckets = buckets
        self.counter = 0
        self.split = 0
        self.merge = 0
        self.min = float('inf')
        self.max= float('-inf')
        self.upper = numbuckets * upper_factor

    def create_histogram(self, attr, gamma, gammam, batchsize, userbucketsize):
        """Reads in data from the file, extending the buckets of the histogram is the values are beyond 
        it, and checks to see if the probability that the counts in the equi-depth buckets are not uniformly 
        distributed is statistically significant (less than alpha) and if so, redistributes the regular buckets."""
        N = 0
        sample = []
        initial = False
        skip = 0
        skipcounter = 0
        with open(self.file) as f:
            reader = csv.reader(f)
            header = reader.next()
            for i in range(0, len(header)):
                header[i] = unicode(header[i], 'utf-8-sig')
            attr_index = header.index(attr)
            for row in reader:
                N += 1
                if float(row[attr_index]) < self.min:
                    self.min = float(row[attr_index])
                if float(row[attr_index]) > self.max:
                    self.max = float(row[attr_index]) 
                if len(set(sample)) < self.numbuckets + 1:
                    sample.append(float(row[attr_index]))
                elif len(set(sample)) == self.numbuckets + 1 and initial == False:
                    self.compute_histogram(N, sample, gamma, gammam)
                    self.plot_histogram(attr, self.buckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    skipcounter += 1
                    self.add_datapoint(float(row[attr_index]), N, sample, attr, gamma, gammam)
                    if skipcounter == skip:
                        sample = self.maintainBackingSample(float(row[attr_index]), sample)
                        skip = self.calculateSkip(len(sample))
                        skipcounter = 0
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_histogram(attr, self.buckets)
                        d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        d.create_distribution(self.buckets)
                        new_buckets = d.return_distribution()
                        self.plot_histogram(attr, new_buckets)
        frequency = []
        for bucket in self.buckets:
            frequency.append(bucket['frequency'])
        cumfreq = np.cumsum(frequency)
        print cumfreq
        #print N
        self.print_buckets()
        realdist = np.array(pd.read_csv(self.file)[attr], dtype=float)
        print stats.kstest(realdist, lambda x: self.callable_cdf(x, cumfreq), N=len(realdist), alternative='two-sided')

    def callable_cdf(self, x, cumfreq):
        values = []
        for value in x:
            v = self.cdf(value, cumfreq)
            if v == None:
                print value, v
                print self.min, self.max
            values.append(v)
        return np.array(values)
    
    def cdf(self, x, cumfreq):
        if x < self.min:
            return 0
        elif x > self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                return cumfreq[i] / cumfreq[len(cumfreq) - 1]

    def compute_histogram(self, N, sample, gamma, gammam):
        l = N / len(sample)
        betaprime = self.numbuckets
        mprime = len(sample)
        c = Counter(sample)
        mostfreq = c.most_common(self.numbuckets + 1)
        mostfreq = sorted(mostfreq, key=lambda x: x[0], reverse=True)
        low2 = min(sample)
        buckets = self.buckets
        high = mostfreq[0][0] + 1
        low = mostfreq[0][0]
        for i in range(0, self.numbuckets - 1):
            if c[mostfreq[i][0]] >= mprime / betaprime:
                buckets[betaprime - 1]['high'] = high
                buckets[betaprime - 1]['low'] = low
                buckets[betaprime - 1]['frequency'] = l * c[mostfreq[i][0]]
                buckets[betaprime - 1]['regular'] = False
                buckets[betaprime - 1]['size'] = high - low
                mprime -= c[mostfreq[i][0]]
                betaprime -= 1
                high = low
                low = mostfreq[i + 1][0]
        for i in range(0, len(mostfreq) - 1):
            for j in range(0, c[mostfreq[i][1]]):
                sample.remove(mostfreq[i][0])
        sample = sorted(sample)
        for i in range(1, betaprime + 1):
            buckets[i - 1]['high'] = sample[i * (mprime // betaprime)]
            buckets[i - 1]['frequency'] = l * (mprime / betaprime)
            buckets[i - 1]['size'] = buckets[i - 1]['high'] - buckets[i - 1]['low']
        for i in range(0, len(buckets)):
            buckets[i]['low'] = low2
            buckets[i]['size'] = buckets[i]['high'] - buckets[i]['low']
            low2 = buckets[i]['high']
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.buckets[0]['low'] = self.min
        self.split = (2 + gamma) * (l * mprime / betaprime)
        self.merge = (l * mprime) / ((2 + gammam) * betaprime)
        self.buckets = buckets

    def calculateSkip(self, n):
        v = random.uniform(0, 1)
        l = 0
        t = n + 1
        num = 1
        quot = num / t
        while quot > v:
            l += 1
            t += 1
            num += 1
            quot = (quot * num) / t
        return l

    def maintainBackingSample(self, value, sample):
        if len(sample) + 1 <= self.upper:
            sample.append(value)
        else:
            rand_index = random.randint(0,len(sample) - 1)
            sample[rand_index] = value
        return sample

    def add_datapoint(self, value, N, sample, attr, gamma, gammam):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
            if self.buckets[0]['frequency'] >= self.split and self.buckets[0]['regular'] == True:
                #print "### BEFORE MINIMUM ###"
                #self.print_buckets()
                self.splitbucket(N, 0, None, 1, sample, gamma, gammam)
                #print "### AFTER MINIMUM ###"
                #self.print_buckets()
                #if self.buckets[0]['low'] != self.min:
                #    print self.min
                #    sys.exit(0)
        elif value >= self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = value + 1 - self.buckets[self.numbuckets - 1]['low']
            if self.buckets[self.numbuckets - 1]['frequency'] >= self.split and self.buckets[self.numbuckets - 1]['regular'] == True:
                self.splitbucket(N, self.numbuckets - 1, self.numbuckets - 2, None, sample, gamma, gammam)
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    if self.buckets[i]['frequency'] >= self.split and self.buckets[i]['regular'] == True:
                        if i == 0:
                            #print "### BEFORE ###"
                            #self.print_buckets()
                            self.splitbucket(N, 0, None, 1, sample, gamma, gammam)
                            #print "### AFTER ###"
                            #self.print_buckets()
                            #if self.buckets[0]['low'] != self.min:
                            #    print self.min
                            #    sys.exit(0)
                        elif i == self.numbuckets - 1:
                            self.splitbucket(N, i, i - 1, None, sample, gamma, gammam)
                        else:
                            self.splitbucket(N, i, i - 1, i + 1, sample, gamma, gammam)
                        

    def splitbucket(self, N, bucketindex, prevbucketindex, afterbucketindex, sample, gamma, gammam):
        s = []
        for i in range(0, len(sample)):
            if sample[i] >= self.buckets[bucketindex]['low'] and sample[i] <= self.buckets[bucketindex]['high']:
                s.append(sample[i])
        m = np.median(s)
        if prevbucketindex != None and m != self.buckets[prevbucketindex]['high'] and m != self.buckets[bucketindex]['high']:
            mergepair_index = self.candidateMergePair()
            #if self.buckets[0]['low'] == self.min:
            #    print "RIGHT!!!!"
            if mergepair_index != None:
                #print mergepair_index
                self.mergebuckets(mergepair_index) # merge the buckets into one bucket
                #if self.buckets[0]['low'] == self.min:
                #    print "RIGHT SO FAR"
                self.splitbucketintwo(bucketindex, sample) # split bucket
                #if self.buckets[0]['low'] == self.min:
                #    print "AND HERE???"
            else:
                #print "COMPUTING HERE"
                self.compute_histogram(N, sample, gamma, gammam)
            #if self.buckets[0]['low'] != self.min:
            #    print "NOT RIGHT FIRST IF"
            #    self.print_buckets()
            #    print self.min
            #    sys.exit(0)
        elif prevbucketindex != None and m == self.buckets[prevbucketindex]['high']:
            c = Counter(sample)
            self.buckets[bucketindex]['frequency'] = self.buckets[prevbucketindex]['frequency'] + self.buckets[bucketindex]['frequency'] - (c[m] * N / len(sample))
            self.buckets[prevbucketindex]['high'] = m
            self.buckets[prevbucketindex]['size'] = m - self.buckets[prevbucketindex]['low']
            self.buckets[prevbucketindex]['frequency'] = c[m] * N / len(sample)
            if self.buckets[bucketindex]['count'] >= self.split:
                self.splitbucket(N, bucketindex, prevbucketindex, afterbucketindex, sample)
            elif self.buckets[bucketindex]['count'] <= self.merge:
                mergepair_index = self.candidateMergePair()
                split_index = self.candidatesplitbucket(self, gamma)
                if mergepair_index != None and split_index != None and split_index > 0:
                    if self.buckets[bucketindex]['high'] == self.buckets[mergepair_index]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index]['frequency']:
                        self.mergebuckets(mergepair_index)
                        after = None
                        if split_index < self.numbuckets - 1:
                            after = split_index + 1
                        self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
                    elif self.buckets[bucketindex]['high'] == self.buckets[mergepair_index + 1]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index + 1]['frequency']:
                        self.mergebuckets(mergepair_index)
                        after = None
                        if split_index < self.numbuckets - 1:
                            after = self.buckets[split_index + 1]
                        self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
                else:
                    self.compute_histogram(N, sample, gamma, gammam)
            #if self.buckets[0]['low'] != self.min:
                #print "NOT RIGHT SECOND IF"
                #self.print_buckets()
                #print self.min
                #sys.exit(0)
        elif m == self.buckets[bucketindex]['high'] and afterbucketindex != None:
            c = Counter(sample)
            self.buckets[bucketindex]['frequency'] = self.buckets[afterbucketindex]['frequency'] + self.buckets[bucketindex]['frequency'] - (c[m] * N / len(sample))
            self.buckets[afterbucketindex]['high'] = m
            self.buckets[afterbucketindex]['size'] = m - self.buckets[afterbucketindex]['low']
            self.buckets[afterbucketindex]['frequency'] = c[m] * N / len(sample)
            if self.buckets[afterbucketindex]['frequency'] <= self.split:
                after = None
                if afterbucketindex < self.numbuckets - 1:
                    after = afterbucketindex + 1
                self.splitbucket(N, afterbucketindex, bucketindex, after, sample)
            elif self.buckets[afterbucketindex]['frequency'] <= self.merge:
                mergepair_index = self.candidateMergePair()
                split_index = self.candidatesplitbucket(self, gamma)
                if mergepair_index != None and split_index != None and split_index > 0:
                    if self.buckets[afterbucketindex]['high'] == self.buckets[mergepair_index]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index]['frequency']:
                        self.mergebuckets(mergepair_index)
                        after = None
                        if split_index < self.numbuckets - 1:
                            after = split_index + 1
                        self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
                    elif self.buckets[bucketindex]['high'] == self.buckets[mergepair_index + 1]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index + 1]['frequency']:
                        self.mergebuckets(mergepair_index)
                        after = None
                        if split_index < self.numbuckets - 1:
                            after = split_index + 1
                        self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
                else:
                    self.compute_histogram(N, sample, gamma, gammam)
            #if self.buckets[0]['low'] != self.min:
            #    print "NOT RIGHT THIRD IF"
            #    self.print_buckets()
            #    print self.min
            #    sys.exit(0)
        

    def splitbucketintwo(self, index, sample):
        """Splits a bucket in the list of buckets of the histogram."""
        s = []
        for i in range(0, len(sample)):
            if sample[i] >= bucket['low'] and sample[i] < bucket['high']:
                s.append(sample[i])
        m = np.median(s)
        s = list(set(s))
        bucket2 = {
            'low': s[s.index(m) + 1],
            'high': self.buckets[index]['high'],
            'size': self.buckets[index]['high'] - s[s.index(m) + 1],
            'frequency': self.split / 2
        }
        #bucket2['high'] = m
        #bucket2['low'] = bucket['low']
        #bucket2['size'] = m - bucket['low']
        self.buckets[index]['high'] = s[s.index(m) + 1]
        self.buckets[index]['frequency'] = self.split / 2
        self.buckets[index]['size'] = self.buckets[index]['high'] - self.buckets[index]['low']
        self.buckets.insert(index + 1, bucket2)
        #bucket2['frequency'] = self.split / 2
        #buckets = []
        #for i in range(0, len(self.buckets)):
        #    if self.buckets[i]['low'] == bucket['low'] and self.buckets[i]['high'] == bucket['high']:
        #        buckets.append(bucket2)
        #        buckets.append(bucket)
        #    else:
        #        buckets.append(self.buckets[i])
        #return buckets

    def mergebuckets(self, index):
        self.buckets[index]['frequency'] = self.buckets[index]['frequency'] + self.buckets[index + 1]['frequency']
        self.buckets[index]['high'] = self.buckets[index + 1]['high']
        self.buckets[index]['size'] = self.buckets[index]['high'] - buckets[index]['low']
        del self.buckets[index + 1]
        return buckets

    def candidateMergePair(self):
        count = 0
        index = None
        for i in range(0, self.numbuckets - 1):
            if self.buckets[i]['regular'] == False and self.buckets[i + 1]['regular'] == False and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.split:
                if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] > count:
                    count = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
                    index = i
            elif self.buckets[i]['regular'] == False and self.buckets[i + 1]['regular'] == True and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.split:
                if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] > count:
                    count = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
                    index = i
            elif self.buckets[i]['regular'] == False and self.buckets[i + 1]['regular'] == True and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.split:
                if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] > count:
                    count = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
                    index = i
        return index

    def candidatesplitbucket(self, gamma):
        count = 0
        index = None
        for i in range(0, self.numbuckets):
            if self.buckets[i]['regular'] == True and self.buckets[i]['frequency'] >= 2(self.merge + 1):
                if self.buckets[i]['frequency'] > count:
                    count = self.buckets[i]['frequency']
                    index = i
            elif self.buckets[i]['regular'] == False and self.buckets[i]['frequency'] <= self.split / (2 + gamma) and self.buckets[i]['frequency'] >= 2(self.merge + 1):
                if self.buckets[i]['frequency'] > count:
                    count = self.buckets[i]['frequency']
                    index = i
        return index

    def plot_histogram(self, attr, buckets):
        """Plots the histogram."""
        bins = []
        frequency = []
        for bucket in buckets:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
        bins.append(bucket['high'])

        frequency = np.array(frequency)
        bins = np.array(bins)

        widths = bins[1:] - bins[:-1]

        plt.bar(bins[:-1], frequency, width=widths, color='#348ABD')

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim(self.min - abs(buckets[0]['size']), self.max + abs(buckets[0]['size']))
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.subplot().set_axis_bgcolor('#E5E5E5');
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Dynamic\ Compressed\ Histogram\ of\ ' + attr + '}$')
        
        with open(self.outputpath + "//data//dc" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//dc" + str(self.counter) + ".jpg")
        plt.clf()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        high = self.buckets[0]['low']
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
            high = self.buckets[i]['high']
         