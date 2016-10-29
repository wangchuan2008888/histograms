"""
It constructs an equi-depth histogram from the dataset given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
#import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import random
import user_distribution
import json
import os
from scipy import stats
from shutil import copyfile

upper_factor = 3

class Equidepth_Histogram(object):

    """
    This class models an instance of an equi-depth histogram, which is a histogram with all the buckets have the same count.
    """

    def __init__(self, f, numbuckets, outputpath):

        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """
        self.outputpath = outputpath
        self.file = f
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, numbuckets):
            buckets.append({
                'low': 0,
                'high': 0, 
                'frequency': 0,
                'size': 0,
            })
        self.buckets = buckets
        self.threshold = None
        self.counter = 0
        self.min = float('inf')
        self.max= float('-inf')
        self.upper = numbuckets * upper_factor

    def zipfdistributiongraph(self, z, l, batchsize, userbucketsize):
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
            self.create_histogram(attr, l, batchsize, userbucketsize)
            f = open(outputpath + "//data//equidepthksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//equidepthzipf.jpg")
        plt.close()

    def create_histogram(self, attr, l, batchsize, userbucketsize):
        """l is a tunable parameter (> -1) which influences the upper thresholder of bucket count for all buckets. The appropriate bucket counter is 
        incremented for every record read in. If a bucket counter reaches threshold, the bucket boundaries are recalculated and the threshold is updated."""
        N = 0
        sample = []
        skipcounter = 0
        skip = 0
        initial = False
        try:
            os.remove(self.outputpath + "//data//equidepthksstats" + ".json")
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
                if len(set(sample)) < self.numbuckets:
                    sample.append(float(row[attr_index]))
                elif len(set(sample)) == self.numbuckets and initial == False:
                    self.computehistogram(sample, N, l)
                    self.plot_histogram(attr, self.buckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    skipcounter += 1
                    self.add_datapoint(float(row[attr_index]), N, sample, attr, l)
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
                        self.compare_histogram(attr, False)
                        f = 0
                        for i in range(len(self.buckets)):
                            f += self.buckets[i]['frequency']
                        print f, N
                        assert np.isclose(f, N)
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
        realdist = []
        with open(self.file, 'r') as f:
            reader = csv.reader(f)
            header = reader.next()
            for i in range(0, len(header)):
                header[i] = unicode(header[i], 'utf-8-sig')
            attr_index = header.index(attr)
            for row in reader:
                realdist.append(float(row[attr_index]))
        #realdist = np.array(pd.read_csv(self.file)[attr], dtype=float)
        if end:
            ksstats = {}
            ksstats['cdfstats'] = stats.kstest(realdist, lambda x: self.callable_cdf(x, cumfreq), N=len(realdist), alternative='two-sided')
            ksstats['linearcdfstats'] = stats.kstest(realdist, lambda x: self.callable_linearcdf(x, cumfreq), N=len(realdist), alternative='two-sided')
            with open(self.outputpath + "//data//equidepthksstats" + ".json", 'a+') as ks:
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
        plt.savefig(self.outputpath + "//img//equidepthcdf" + str(self.counter) + ".jpg")  
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

    def add_datapoint(self, value, N, sample, attr, l):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            b = {
                'low': value,
                'high': self.buckets[0]['low'],
                'frequency': 1,
                'size': self.buckets[0]['low'] - value
            }
            self.buckets.insert(0, b)
            index = self.mergebucketPair(b)
            if index != None:
                self.mergebuckets(index, index + 1)
            else:
                self.buckets[0]['frequency'] += self.buckets[1]['frequency']
                self.buckets[0]['high'] = self.buckets[1]['high']
                self.buckets[0]['size'] = self.buckets[0]['high'] - value
                del self.buckets[1]
            if self.buckets[0]['frequency'] >= self.threshold:
                self.thresholdReached(self.buckets[0], N, sample, attr, l)
        elif value > self.buckets[self.numbuckets - 1]['high']:
            b = {
                'low': self.buckets[self.numbuckets - 1]['high'],
                'high': value + 1,
                'frequency': 1,
                'size': value - self.buckets[self.numbuckets - 1]['high']
            }
            self.buckets.append(b)
            index = self.mergebucketPair(b)
            if index != None:
                self.mergebuckets(index, index + 1)
            else:
                self.buckets[self.numbuckets - 1]['high'] = value + 1
                self.buckets[self.numbuckets - 1]['frequency'] += 1
                self.buckets[self.numbuckets - 1]['size'] = value + 1 - self.buckets[self.numbuckets - 1]['low']
                del self.buckets[self.numbuckets]
            if self.buckets[self.numbuckets - 1]['frequency'] >= self.threshold:
                self.thresholdReached(self.buckets[self.numbuckets - 1], N, sample, attr, l)
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    if self.buckets[i]['frequency'] >= self.threshold:
                        self.thresholdReached(self.buckets[i], N, sample, attr, l)

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

    def mergebucketPair(self, bucket):
        minimum = float('inf')
        index = None
        for i in range(0, len(self.buckets) - 1):
            if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.threshold and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < minimum:
                if self.buckets[i]['low'] != bucket['low'] and self.buckets[i + 1]['low'] != bucket['low']:
                    minimum = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
                    index = i
        return index


    def thresholdReached(self, bucket, N, sample, attr, l):
        #print "threshold reached"
        index = self.mergebucketPair(bucket)
        if index != None:
            self.mergebuckets(index, index + 1)
            splitindex = None
            for i in range(self.numbuckets - 1):
                if self.buckets[i]['low'] == bucket['low'] and self.buckets[i]['high'] == bucket['high']:
                    splitindex = i
            self.splitbucket(splitindex, bucket['low'], bucket['high'], sample)
        else:
            self.computehistogram(sample, N, l)

    def computehistogram(self, sample, N, l):
        sorted_sample = sorted(list(set(sample)), key=float)
        #sorted(sample, key=float)
        frac = len(sorted_sample) / self.numbuckets
        equal = N / self.numbuckets
        for i in range(0, self.numbuckets):
            index = int(round((i + 1) * frac))
            self.buckets[i]['low'] = sorted_sample[int(round(i * frac))]
            if i == self.numbuckets - 1:
                self.buckets[i]['high'] = sorted_sample[len(sorted_sample) - 1] + 1
            else:
                self.buckets[i]['high'] = sorted_sample[index]
            self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
            self.buckets[i]['frequency'] = (i * equal) - ((i - 1) * equal)
        self.buckets[0]['low'] = self.min
        self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.buckets[self.numbuckets - 1]['size'] = self.max + 1 - self.buckets[self.numbuckets - 1]['low']
        self.threshold = (2 + l) * (N / self.numbuckets)

    def mergebuckets(self, b1, b2):
        """Merging two buckets into one bucket in the list of buckets."""
        self.buckets[b1]['high'] = self.buckets[b2]['high']
        self.buckets[b1]['size'] = self.buckets[b1]['high'] - self.buckets[b1]['low']
        self.buckets[b1]['frequency'] += self.buckets[b2]['frequency']
        del self.buckets[b2]

    def splitbucket(self, index, low, high, sample):
        """Splits a bucket in the list of buckets of the histogram."""
        s = []
        sorted_sample = sorted(sample, key=float)
        for i in range(0, len(sorted_sample)):
            if sorted_sample[i] >= low and sorted_sample[i] < high:
                s.append(sorted_sample[i])
        m = np.median(s)
        medianhigh = None
        if len(sorted_sample) % 2 != 0:
            medianindex = sorted_sample.index(m)
            medianhigh = sorted_sample[medianindex + 1]
        else:
            medianhigh = m
        self.buckets[index]['high'] = medianhigh
        self.buckets[index]['size'] = medianhigh - self.buckets[index]['low']
        self.buckets[index]['frequency'] = self.threshold / 2
        b = {
            'low': medianhigh,
            'high': high,
            'size': high - medianhigh,
            'frequency': self.threshold / 2
        }
        self.buckets.insert(index + 1, b)

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
        plt.subplot().set_axis_bgcolor('#E5E5E5');
        plt.xlabel(attr)
        plt.ylabel('Frequency')
        plt.title(r'$\mathrm{Equi-Depth\ Histogram\ of\ ' + attr + '}$')
                
        with open(self.outputpath + "//data//equidepth" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//equidepth" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""        
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"