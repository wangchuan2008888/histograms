"""
It constructs a spline histogram from the dataset given.

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
import heapq
import random
import user_distribution
import json
import os
from scipy import stats

upper_factor = 3

class PriorityQueueSet(object):

    """
    Combined priority queue and set data structure.

    Acts like a priority queue, except that its items are guaranteed to be
    unique. Provides O(1) membership test, O(log N) insertion and O(log N)
    removal of the smallest item.

    Important: the items of this data structure must be both comparable and
    hashable (i.e. must implement __cmp__ and __hash__). This is true of
    Python's built-in objects, but you should implement those methods if you
    want to use the data structure for custom objects.


    from: 
    http://stackoverflow.com/questions/407734/a-generic-priority-queue-for-python
    """

    def __init__(self, items=[]):
        """
        Create a new PriorityQueueSet.

        Arguments:
            items (list): An initial item list - it can be unsorted and
                non-unique. The data structure will be created in O(N).
        """
        self.set = dict((item, True) for item in items)
        self.heap = self.set.keys()
        heapq.heapify(self.heap)

    def has_item(self, item):
        """Check if ``item`` exists in the queue."""
        return item in self.set

    def pop_smallest(self):
        """Remove and return the smallest item from the queue."""
        smallest = heapq.heappop(self.heap)
        del self.set[smallest]
        return smallest

    def add(self, item):
        """Add ``item`` to the queue if doesn't already exist."""
        if item not in self.set:
            self.set[item] = True
            heapq.heappush(self.heap, item)

    def remove(self, item):
        """Removes ''item'' from the heap if it exists."""
        if self.has_item(item):
            del self.set[item]
            self.heap = self.set.keys()
            heapq.heapify(self.heap)
            #self.heap = list(set(self.set.keys()) - set(item))
            #heapq.heapify(self.heap)

class Spline_Histogram(object):

    """
    This creates an instance of a histogram for a dataset. It reads the dataset
    in batches and computes the approximate histogram, plotting the histogram with 
    every batch. The histogram approximates the frequency in the buckets with a 
    linear spline function and chooses the buckets based on 
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
        for i in range(0, numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'size': 0,
                'frequency': 0,
                'ff': 0,
                'vv': 0,
                'vf': 0,
                'v': [0, 0, 0]
            })
        self.buckets = buckets
        self.counter = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.upper = numbuckets * upper_factor
    
    def create_histogram(self, attr, batchsize, userbucketsize):
        """Reads in records from the file, computing the initial histogram and after each batch by using a 
        greedy merge algorithm that creates N / 2 buckets and continually merges buckets with the smallest 
        error until there are numbuckets left."""
        N = 0
        sample = []
        initial = False
        skip = 0
        skipcounter = 0
        try:
            os.remove(self.outputpath + "//data//splineksstats" + ".json")
        except OSError:
            pass
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
                if len(set(sample)) < self.numbuckets * 2:
                    sample.append(float(row[attr_index]))
                elif len(set(sample)) == self.numbuckets * 2 and initial == False:
                    self.compute_histogram(sample, N)
                    self.plot_histogram(attr, self.buckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    skipcounter += 1
                    self.add_datapoint(float(row[attr_index]))
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
                        self.compute_histogram(sample, N)
                        self.compare_histogram(attr, False)
        self.compare_histogram(attr, True)

    def compare_histogram(self, attr, end):
        frequency = []
        binedges = []
        for bucket in self.buckets:
            frequency.append(bucket['frequency'])
            binedges.append(bucket['low'])
        binedges.append(bucket['high'])
        cumfreq = np.cumsum(frequency)
        realdist = np.array(pd.read_csv(self.file)[attr], dtype=float)
        if end:
            ksstats = {}
            ksstats['cdfstats'] = stats.kstest(realdist, lambda x: self.callable_cdf(x, cumfreq), N=len(realdist), alternative='two-sided')
            ksstats['linearcdfstats'] = stats.kstest(realdist, lambda x: self.callable_linearcdf(x, cumfreq), N=len(realdist), alternative='two-sided')
            with open(self.outputpath + "//data//splineksstats" + ".json", 'a+') as ks:
                json.dump(ksstats, ks)
                ks.write('\n')
            self.counter += 1        
        sorted_data = np.sort(realdist)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        plt.grid(True)
        plt.plot(sorted_data, yvals)
        step = [0]
        step.extend(cumfreq / cumfreq[len(cumfreq) - 1])
        plt.step(binedges[0:], step)
        plt.plot(binedges[0:], step)
        plt.legend(['CDF of real data', 'CDF of histogram', 'CDF of linear approx'], loc='lower right')
        plt.savefig(self.outputpath + "//img//splinecdf" + str(self.counter) + ".jpg")  
        self.counter += 1
        plt.close()

    def callable_cdf(self, x, cumfreq):
        values = []
        for value in x:
            v = self.cdf(value, cumfreq)
            if v == None:
                print value, v
                print self.min, self.max
            values.append(v)
        return np.array(values)

    def callable_linearcdf(self, x, cumfreq):
        values = []
        for value in x:
            values.append(self.linear_cdf(value, cumfreq))
        return np.array(values)
    
    def cdf(self, x, cumfreq):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                return cumfreq[i] / cumfreq[len(cumfreq) - 1]

    def linear_cdf(self, x, cumfreq):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                approx = None
                percentage = (x - self.buckets[i]['low']) / self.buckets[i]['size']
                if i > 0:
                    approx = percentage + cumfreq[i - 1]
                else:
                    approx = percentage * cumfreq[i]
                return approx / cumfreq[len(cumfreq) - 1] 

    def add_datapoint(self, value):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1

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


    def compute_histogram(self, sample, N):
        """Computes the histogram using a greedy merge algorithm that creates N / 2 buckets and continually 
        merges buckets with the smallest error until there are numbuckets left."""
        n = len(sample)
        sample = sorted(sample, key=float)
        buckets = []
        c = Counter(sample)
        for i in range(0, n - 1):
            if 2 * (i + 1) == n:
                sample.append(sample[n - 1] + 1)
            elif 2 * (i + 1) > n: # to fix the indexing issue
                break
            buckets.append({
                'low': sample[(2 * i)],
                'high': sample[(2 * (i + 1))],
                'size': sample[(2 * (i + 1))] - sample[(2 * i)],
                'frequency': (c[sample[(2 * i)]] + c[sample[(2 * i) + 1]]) * N / len(sample),
                'ff': np.power(c[sample[(2 * i)]], 2) + np.power(c[sample[(2 * i) + 1]], 2),
                'vv': np.power(sample[(2 * i)], 2) + np.power(sample[(2 * i) + 1], 2),
                'vf': (c[sample[(2 * i) + 1]] * sample[(2 * i) + 1]) + (c[sample[(2 * i)]] * sample[(2 * i)]),
                'v': [sample[(2 * i) + 1] + sample[2 * i], (sample[(2 * i) + 1] + sample[2 * i]), c[sample[(2 * i) + 1]] + c[sample[2 * i]]]
            })
            if 2 * (i + 1) >= n:
                break
        q = PriorityQueueSet()
        b = {}
        for i in range(0, len(buckets) - 1):
            if i < len(buckets) - 1:
                error = self.spline_error(buckets[i]['low'], buckets[i + 1]['high'], sample, buckets[i], buckets[i + 1])
                q.add(error)
                b[error] = [i, i + 1]
        while len(buckets) > self.numbuckets:
            minerror = q.pop_smallest()
            if b[minerror][0] > 0:
                if b[minerror][0] > len(buckets):
                    print len(buckets), b[minerror][0]
                else:
                    leftbucket = buckets[b[minerror][0] - 1]
                    lefterror = self.spline_error(leftbucket['low'], buckets[b[minerror][0]]['high'], sample, leftbucket, buckets[b[minerror][0]])
                    q.remove(lefterror)
                    if b.has_key(lefterror):
                        del b[lefterror]
            if b[minerror][1] < len(buckets) - 1:
                rightbucket = buckets[b[minerror][1] + 1]
                righterror = self.spline_error(buckets[b[minerror][1]]['low'], rightbucket['high'], sample, buckets[b[minerror][1]], rightbucket)
                q.remove(righterror)
                if b.has_key(righterror):
                    del b[righterror]
            left = b[minerror][0]
            right = b[minerror][1]
            buckets = self.mergebuckets(buckets, buckets[left], buckets[right])
            del b[minerror]
            if left > 0:
                error = self.spline_error(buckets[left - 1]['low'], buckets[left]['high'], sample, buckets[left - 1], buckets[left])
                q.add(error)
                b[error] = [left - 1, left]
            if right < len(buckets) - 1:
                error = self.spline_error(buckets[right]['low'], buckets[right + 1]['high'], sample, buckets[right], buckets[right + 1])
                q.add(error)
                b[error] = [right, right + 1]
        buckets[0]['low'] = self.min
        buckets[0]['size'] = buckets[0]['high'] - buckets[0]['low']
        buckets[len(buckets) - 1]['high'] = self.max + 1
        buckets[len(buckets) - 1]['size'] = buckets[len(buckets) - 1]['high'] - buckets[len(buckets) - 1]['low']
        self.buckets = buckets


    def mergebuckets(self, buckets, bucket1, bucket2):
        "This merges the two buckets (bucket1, bucket2) in buckets."""
        mergedbucket = {
            'low': bucket1['low'],
            'high': bucket2['high'],
            'size': bucket2['high'] - bucket1['low'],
            'frequency': bucket1['frequency'] + bucket2['frequency'],
            'ff': bucket1['ff'] + bucket2['ff'],
            'vv': bucket1['vv'] + bucket2['vv'],
            'vf': bucket1['vf'] + bucket2['vf'],
            'v': [bucket1['v'][0] + bucket2['v'][0], np.average([bucket1['v'][1], bucket2['v'][1]]), np.average([bucket1['v'][2], bucket2['v'][2]])]
        }
        b = []
        for i in range(0, len(buckets)):
            if buckets[i]['low'] == bucket1['low'] and buckets[i]['high'] == bucket1['high']:
                b.append(mergedbucket)
            elif buckets[i]['low'] == bucket2['low'] and buckets[i]['high'] == bucket2['high']:
                pass
            else:
                b.append(buckets[i])
        return b


    def correlation(self, a, c, sample, bucket1, bucket2):
        """Calculates the correlation between two buckets than span the range a-c."""
        sample = list(set(sample))
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        avgv = (bucket1['v'][1] + bucket2['v'][1]) / 2
        avgf = (bucket2['v'][2] + bucket2['v'][2]) / 2
        c = Counter(sample)
        for i in range(0, len(sample)):
            if sample[i] >= a and sample[i] < c:
                if sample[i] < bucket1['high']:
                    numerator += (sample[i] - avgv) * (c[sample[i]] * avgf)
                    denominator1 += np.power(sample[i] - avgv, 2)
                    denominator2 += np.power(c[sample[i]] - avgf, 2)
                else:
                    numerator += (sample[i] - avgv) * (c[sample[i]] * avgf)
                    denominator1 += np.power(sample[i] - avgv, 2)
                    denominator2 += np.power(c[sample[i]] - avgf, 2)
        return numerator / (np.power(denominator1, 0.5) * np.power(denominator2, 0.5))

    def spline_error(self, a, c, sample, bucket1, bucket2):
        """Calculates the spline error between two buckets that span the range a-c."""
        corr = self.correlation(a, c, sample, bucket1, bucket2)
        error = 1 - np.power(corr, 2)
        error2 = bucket1['ff'] + bucket2['ff']
        error2 *= bucket1['frequency'] + bucket2['frequency']
        error2 += c - a
        error2 *= np.average([bucket1['v'][2], bucket2['v'][2]])
        return error * error2

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

        plt.bar(bins[:-1], frequency, width=widths, edgecolor=['black'], color='#348ABD')

        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim(self.min - abs(buckets[0]['size']), self.max + abs(buckets[0]['size']))
        axes.set_ylim([0, max(frequency) + max(frequency) / 2])
        plt.subplot().set_axis_bgcolor('#E5E5E5')
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Spline\ Histogram\ of\ ' + attr + '}$')
        
        with open(self.outputpath + "//data//spline" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//spline" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"