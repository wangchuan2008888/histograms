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
from shutil import copyfile
import operator

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

    def zipfdistributiongraph(self, z, batchsize, userbucketsize):
        ksstatistics = []
        zipfparameter = []
        path = self.outputpath
        for parameter in z:
            self.counter = 0
            self.min = float('inf')
            self.max= float('-inf')
            print "zipf parameter" + str(parameter)
            zipfparameter.append(parameter)
            attr = 'zipf' + str(parameter)
            outputpath = 'output//' + attr + '//' + str(batchsize) + '_' + str(self.numbuckets) + '_' + str(userbucketsize)
            if not os.path.exists(outputpath + '//img'):
                os.makedirs(outputpath + '//img')
            if not os.path.exists(outputpath + '//data'):
                os.makedirs(outputpath + '//data')
            copyfile('template.html', outputpath + '//template.html')
            copyfile('d3.html', outputpath + '//d3.html')
            copyfile('template.html', outputpath + '//template.html')
            self.outputpath = outputpath
            self.create_histogram(attr, batchsize, userbucketsize)
            f = open(outputpath + "//data//splineksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//splinezipf.jpg")
        plt.close()
    
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
                    self.add_datapoint(float(row[attr_index]), sample)
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
                else:
                    print("ERROR: There are not enough unique values for the number of specified buckets.")
        self.compare_histogram(attr, False)

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

    def add_datapoint(self, value, sample):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            bucket = {
                'ff': 1,
                'vv': np.power(value, 2),
                'vf': value,
                'v': [value, value, 1],
                'low': value,
                'high': self.buckets[0]['low'],
                'frequency': 1,
                'size': value + 1 - self.buckets[self.numbuckets - 1]['high']
            }
            self.mergesmallest(sample)
            self.buckets.append(bucket) # borrow one bucket
            #print "new bucket: " + str(bucket['low']) + ", " + str(bucket['high']) + ", " + str(len(self.buckets))
        elif value > self.buckets[self.numbuckets - 1]['high']:
            bucket = {
                'ff': 1,
                'vv': np.power(value, 2),
                'vf': value,
                'v': [value, value, 1],
                'low': self.buckets[self.numbuckets - 1]['high'],
                'high': value + 1,
                'frequency': 1,
                'size': value + 1 - self.buckets[self.numbuckets - 1]['high']
            }
            self.mergesmallest(sample)
            self.buckets.append(bucket)
            #print "new bucket: " + str(bucket['low']) + ", " + str(bucket['high']) + ", " + str(len(self.buckets))
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
        if self.min not in sample:
            sample.append(self.min)
        if self.max not in sample:
            sample.append(self.max)
        return sample


    def compute_histogram(self, sample, N):
        """Computes the histogram using a greedy merge algorithm that creates N / 2 buckets and continually 
        merges buckets with the smallest error until there are numbuckets left."""
        n = len(sample)
        sample = sorted(list(set(sample)), key=float)
        #sorted(sample, key=float)
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
        self.buckets = buckets
        for i in range(0, len(self.buckets) - 1):
            if i < len(buckets) - 1:
                error = self.spline_error(self.buckets[i]['low'], self.buckets[i + 1]['high'], sample, self.buckets[i], self.buckets[i + 1])
                q.add(error)
                b[error] = i
        while len(buckets) > self.numbuckets:
            # NEEDS TO BE FIXED
            minerror = q.pop_smallest()
            left = b[minerror]
            right = left + 1
            lefterror = None
            righterror = None
            if left > 0 and right < len(self.buckets) - 1:
                lefterror = self.spline_error(self.buckets[left - 1]['low'], self.buckets[left]['high'], sample, self.buckets[left - 1], self.buckets[left])
                righterror = self.spline_error(self.buckets[right]['low'], self.buckets[right + 1]['high'], sample, self.buckets[right], self.buckets[right + 1])
                #print "left: " + str(left)
                #print "right: " + str(right)
                if b.has_key(lefterror):
                    del b[lefterror]
                if b.has_key(righterror):
                    del b[righterror]
                q.remove(lefterror)
                q.remove(righterror)
                self.mergebuckets(left, right)
                lefterror = self.spline_error(self.buckets[left - 1]['low'], self.buckets[left]['high'], sample, self.buckets[left - 1], self.buckets[left])
                b[lefterror] = left - 1
                q.add(lefterror)
                righterror = self.spline_error(self.buckets[left]['low'], self.buckets[left + 1]['high'], sample, self.buckets[left], self.buckets[left + 1])
                b[righterror] = left
                q.add(righterror)
                b = self.adjustindexes(b, left)
            elif left == 0:
                righterror = self.spline_error(self.buckets[right]['low'], self.buckets[right + 1]['high'], sample, self.buckets[right], self.buckets[right + 1])
                del b[righterror]
                q.remove(righterror)
                self.mergebuckets(left, right)
                righterror = self.spline_error(self.buckets[left]['low'], self.buckets[left + 1]['high'], sample, self.buckets[left], self.buckets[left + 1])
                b[righterror] = left
                q.add(righterror)
                b = self.adjustindexes(b, left)
            elif right == len(self.buckets) - 1:
                lefterror = self.spline_error(self.buckets[left - 1]['low'], self.buckets[left]['high'], sample, self.buckets[left - 1], self.buckets[left])
                del b[lefterror]
                q.remove(lefterror)
                self.mergebuckets(left, right)
                lefterror = self.spline_error(self.buckets[left - 1]['low'], self.buckets[left]['high'], sample, self.buckets[left - 1], self.buckets[left])
                b[lefterror] = left - 1
                q.add(lefterror)
                b = self.adjustindexes(b, left)
        self.buckets[0]['low'] = self.min
        self.buckets[0]['size'] = buckets[0]['high'] - buckets[0]['low']
        self.buckets[len(buckets) - 1]['high'] = self.max + 1
        self.buckets[len(buckets) - 1]['size'] = buckets[len(buckets) - 1]['high'] - buckets[len(buckets) - 1]['low']

    def adjustindexes(self, bucketindexes, index):
        #b = dict(sorted(bucketindexes.items(), key=operator.itemgetter(1)))
        for k,v in bucketindexes.items():
            if v > index:
                #print k,v
                #v -= 1
                #print k,v
                bucketindexes[k] -= 1
        return bucketindexes

    def mergebuckets(self, b1, b2):
        "This merges the two buckets at indexes b1 and b2 in self.buckets."""
        self.buckets[b1]['frequency'] += self.buckets[b2]['frequency']
        self.buckets[b1]['high'] = self.buckets[b2]['high']
        self.buckets[b1]['size'] = self.buckets[b1]['high'] - self.buckets[b1]['low']
        self.buckets[b1]['ff'] += self.buckets[b2]['ff']
        self.buckets[b1]['vv'] += self.buckets[b2]['vv']
        self.buckets[b1]['vf'] += self.buckets[b2]['vf']
        self.buckets[b1]['v'] = [self.buckets[b1]['v'][0] + self.buckets[b2]['v'][0], np.average([self.buckets[b1]['v'][1], self.buckets[b2]['v'][1]]), np.average([self.buckets[b1]['v'][2], self.buckets[b2]['v'][2]])]
        del self.buckets[b2]

        # mergedbucket = {
        #     'low': bucket1['low'],
        #     'high': bucket2['high'],
        #     'size': bucket2['high'] - bucket1['low'],
        #     'frequency': bucket1['frequency'] + bucket2['frequency'],
        #     'ff': bucket1['ff'] + bucket2['ff'],
        #     'vv': bucket1['vv'] + bucket2['vv'],
        #     'vf': bucket1['vf'] + bucket2['vf'],
        #     'v': [bucket1['v'][0] + bucket2['v'][0], np.average([bucket1['v'][1], bucket2['v'][1]]), np.average([bucket1['v'][2], bucket2['v'][2]])]
        # }
        # b = []
        # for i in range(0, len(buckets)):
        #     if buckets[i]['low'] == bucket1['low'] and buckets[i]['high'] == bucket1['high']:
        #         b.append(mergedbucket)
        #     elif buckets[i]['low'] == bucket2['low'] and buckets[i]['high'] == bucket2['high']:
        #         pass
        #     else:
        #         b.append(buckets[i])
        # return b


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

    def mergesmallest(self, sample):
        minimum = float("inf")
        index = None
        for i in range(self.numbuckets - 1):
            err = self.spline_error(self.buckets[i]['low'], self.buckets[i + 1]['high'], sample, self.buckets[i], self.buckets[i + 1])
            if err < minimum:
                minimum = err
                index = i
        self.mergebuckets(index, index + 1)

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