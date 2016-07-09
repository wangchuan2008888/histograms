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

class DVO_Histogram(object):

    def __init__(self, fil, numbuckets):
        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'size': 0,
                'leftcounter': 0,
                'rightcounter': 0
            })
        self.buckets = buckets

    # implements a dynamic v-optimal histogram while reading the file
    def create_dc_histogram(self, attr, alpha, batchsize):
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
                        self.buckets[i]['leftcounter'] = c[buckets[i]]
                        self.buckets[i]['rightcounter'] = c[buckets[i]]
                        self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                        if buckets[i] not in self.buckets[i]['unique']:
                            self.buckets[i]['unique'].append(buckets[i])
                elif len(set(sample)) > self.numbuckets:
                    print "number read in: " + str(N)
                    self.add_datapoint(float(row[attr_index]))

    # this method adds data points to the histograms, adjusting the end bucket partitions if necessary
    def add_datapoint(self, value):
        if value > self.buckets[self.numbuckets]['high']:
            bucket = {
                'low': self.buckets[self.numbuckets]['high'],
                'high': value + 1,
                'leftcounter': 1,
                'rightcounter': 1,
                'size': value + 1 - self.buckets[self.numbuckets]['high']
            } # borrow one bucket
            index = self.findBestToMerge()
            self.mergebuckets(self.buckets[index], self.buckets[index + 1])
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    if value < self.buckets[i]['low'] + (self.buckets[i]['size'] / 2):
                        self.buckets[i]['leftcounter'] += 1
                    else:
                        self.buckets[i]['rightcounter'] += 1
            s = self.buckets[self.findBestToSplit()]
            mindex = self.findBestToMerge()
            if self.bucketError(s) > self.adjacentbucketsError(self.buckets[mindex], self.buckets[mindex + 1]):
                # split s
                # merge m and m.next
                self.splitbucket(s)
                self.merge(self.buckets[mindex], self.buckets[mindex + 1])

    # merging two buckets into one bucket in the list of buckets
    def mergebuckets(self, bucket1, bucket2):
        mergedbucket = {
            'low': bucket1['low'],
            'high': bucket2['high'],
            'size': bucket2['high'] - bucket1['low'],
            'leftcounter': bucket1['leftcounter'] + bucket1['rightcounter'],
            'rightcounter': bucket2['leftcounter'] + bucket2['rightcounter']
        }
        buckets = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]]['low'] < bucket1['low'] or self.buckets[i]['low'] >= bucket2['high']:
                buckets.append(self.buckets[i])
            elif self.buckets[i]['low'] == bucket1['low']:
                buckets.append(mergedbucket)
            elif self.buckets[i]['low'] == bucket2['low']:
                pass

    def splitbucket(self, bucket):
        bucket1 = {
            'low': bucket['low'],
            'high': (bucket['size'] / 2) + bucket['low'],
            'size': (bucket['size'] / 2) + bucket['low'] - bucket['low'],
            'leftcounter': bucket['leftcounter'],
            'rightcounter': bucket['leftcounter']
        }
        bucket2 = {
            'low': bucket1['low'],
            'high': bucket['high'],
            'size': bucket1['size'],
            'leftcounter': bucket['rightcounter'],
            'rightcounter': bucket['rightcounter']
        }
        buckets = []
        for i in range(0, self.numbuckets):
            if self.buckets[i]]['low'] < bucket['low'] or self.buckets[i]['low'] >= bucket['high']:
                buckets.append(self.buckets[i])
            else:
                buckets.append(bucket1)
                buckets.append(bucket2)


    # calculates the error of a single bucket
    def bucketError(self, bucket):
        average = (bucket['leftcounter'] + bucket['rightcounter']) / 2
        lefterror = np.power(bucket['leftcounter'] - average, 2)
        righterror = np.power(bucket['rightcounter'] - average, 2)
        return lefterror + righterror

    # calculates the error of two adjacent buckets
    def adjacentbucketsError(self, bucket1, bucket2):
        return self.bucketError(bucket1) + self.bucketError(bucket2)

    # returns the index of a bucket such that combined  with its successor is smallest among all the pairs.
    def findBestToMerge(self):
        minimum = float('inf')
        index = None
        for i in range(0, self.numbuckets - 1):
            error = self.adjacentbucketsError(self.buckets[i], self.buckets[i + 1])
            if error < minimum:
                minimum = error
                index = i
        return minimum, index
    
    # returns the index of a bucket that has the highest error
    def findBestToSplit(self):
        maximum = float('-inf')
        index = None
        for i in range(0, self.numbuckets):
            error = self.bucketError(self.buckets[i])
            if error > maximum:
                maximum = error
                index = i
        return maximum, index
    