"""
It constructs a dynamic compressed histogram from the sample given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from collections import Counter
import user_distribution
import json
import os
from scipy import stats
from shutil import copyfile
import copy

import sys

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
        self.singular = []
        self.regular = []
        self.counter = 0
        self.split = 0
        self.merge = 0
        self.min = float('inf')
        self.max= float('-inf')
        #self.upper = numbuckets * upper_factor

    def zipfdistributiongraph(self, z, gamma, gammam, batchsize, userbucketsize):
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
            self.create_histogram(attr, gamma, gammam, batchsize, userbucketsize)
            f = open(outputpath + "//data//dcksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//dczipf.jpg")
        plt.close()

    def create_histogram(self, attr, batchsize, userbucketsize):
        """Reads in data from the file, extending the buckets of the histogram is the values are beyond 
        it, and checks to see if the probability that the counts in the equi-depth buckets are not uniformly 
        distributed is statistically significant (less than alpha) and if so, redistributes the regular buckets."""
        N = 0
        sample = []
        initial = False
        #skip = 0
        #skipcounter = 0
        try:
            os.remove(self.outputpath + "//data//dcksstats" + ".json")
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
                if len(set(sample)) == self.numbuckets and initial == False:
                    self.compute_histogram(N, sample)

                    totalbuckets = copy.deepcopy(self.regular)
                    for i in range(len(self.singular)):
                        bucket = self.singular[i].copy()
                        for j in range(len(totalbuckets)):
                            if bucket['size'] == 0:
                                break
                            elif totalbuckets[j]['low'] <= bucket['low'] < totalbuckets[j]['high'] and totalbuckets[j][
                                'low'] < \
                                    bucket['high'] <= totalbuckets[j]['high']:
                                totalbuckets[j]['frequency'] += bucket['frequency']
                                bucket['size'] = 0
                                # if the bucket is completely contained within another regular bucket, then split that bucket evenly
                                # might need to come up with something different because of the last condition
                            elif totalbuckets[j]['low'] <= bucket['low'] < totalbuckets[j]['high'] and bucket['high'] > \
                                    totalbuckets[j]['high']:
                                # the case when the left boundary overlaps with a bucket but only part of the bucket will be added to this bucket
                                frac = (totalbuckets[j]['high'] - bucket['low']) / bucket['size']
                                perc = frac * bucket['frequency']
                                totalbuckets[j]['frequency'] += perc
                                bucket['frequency'] -= perc
                                bucket['low'] = totalbuckets[j]['high']
                                bucket['size'] = bucket['high'] - bucket['low']

                    self.plot_histogram(attr, totalbuckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(totalbuckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    #skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    #skipcounter += 1
                    self.add_datapoint(float(row[attr_index]), N, sample, attr)
                    #if skipcounter == skip:
                    #    sample = self.maintainBackingSample(float(row[attr_index]), sample)
                    #    skip = self.calculateSkip(len(sample))
                    #    skipcounter = 0
                    if N % batchsize == 0:
                        f = 0
                        for i in range(len(self.regular)):
                            f += self.regular[i]['frequency']
                        for i in range(len(self.singular)):
                            f += self.singular[i]['frequency']
                        assert np.isclose(f, N)
                        print "number read in: " + str(N)

                        totalbuckets = copy.deepcopy(self.regular)
                        for i in range(len(self.singular)):
                            bucket = self.singular[i].copy()
                            for j in range(len(totalbuckets)):
                                if bucket['size'] == 0:
                                    break
                                elif totalbuckets[i]['low'] <= bucket['low'] < totalbuckets[i]['high'] and \
                                                        totalbuckets[i]['low'] < \
                                                        bucket['high'] <= totalbuckets[i]['high']:
                                    totalbuckets[i]['frequency'] += bucket['frequency']
                                    bucket['size'] = 0
                                    # if the bucket is completely contained within another regular bucket, then split that bucket evenly
                                    # might need to come up with something different because of the last condition
                                elif totalbuckets[i]['low'] <= bucket['low'] < totalbuckets[i]['high'] and bucket[
                                    'high'] > \
                                        totalbuckets[i]['high']:
                                    # the case when the left boundary overlaps with a bucket but only part of the bucket will be added to this bucket
                                    frac = (totalbuckets[i]['high'] - bucket['low']) / bucket['size']
                                    perc = frac * bucket['frequency']
                                    totalbuckets[i]['frequency'] += perc
                                    bucket['frequency'] -= perc
                                    bucket['low'] = totalbuckets[i]['high']
                                    bucket['size'] = bucket['high'] - bucket['low']

                        self.plot_histogram(attr, totalbuckets)
                        d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        d.create_distribution(totalbuckets)
                        new_buckets = d.return_distribution()
                        self.plot_histogram(attr, new_buckets)
                        self.compare_histogram(attr, False, totalbuckets)
                        f = 0
                        for i in range(len(self.regular)):
                            f += self.regular[i]['frequency']
                        for i in range(len(self.singular)):
                            f += self.singular[i]['frequency']
                        assert np.isclose(f, N)
            if len(set(sample)) < self.numbuckets:
                print("ERROR: There are not enough unique values for the number of specified buckets.")
        #self.compare_histogram(attr, False)

    def compare_histogram(self, attr, end, buckets):
        frequency = []
        binedges = []
        for bucket in buckets:
            frequency.append(bucket['frequency'])
            binedges.append(bucket['low'])
        binedges.append(bucket['high'])
        cumfreq = np.cumsum(frequency)
        realdist = np.array(pd.read_csv(self.file)[attr], dtype=float)
        if end:
            ksstats = {}
            ksstats['cdfstats'] = stats.kstest(realdist, lambda x: self.callable_cdf(x, cumfreq, buckets), N=len(realdist), alternative='two-sided')
            ksstats['linearcdfstats'] = stats.kstest(realdist, lambda x: self.callable_linearcdf(x, cumfreq, buckets), N=len(realdist), alternative='two-sided')
            with open(self.outputpath + "//data//dcksstats" + ".json", 'a+') as ks:
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
        plt.savefig(self.outputpath + "//img//dccdf" + str(self.counter) + ".jpg")
        self.counter += 1
        plt.close()

    def callable_cdf(self, x, cumfreq, buckets):
        values = []
        for value in x:
            v = self.cdf(value, cumfreq, buckets)
            if v == None:
                print value, v
                print self.min, self.max
            values.append(v)
        return np.array(values)

    def callable_linearcdf(self, x, cumfreq, buckets):
        values = []
        for value in x:
            values.append(self.linear_cdf(value, cumfreq, buckets))
        return np.array(values)
    
    def cdf(self, x, cumfreq, buckets):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        #for i in range(0, self.numbuckets):
        for i in range(len(buckets)):
            if x >= buckets[i]['low'] and x < buckets[i]['high']:
                return cumfreq[i] / cumfreq[len(cumfreq) - 1]

    def linear_cdf(self, x, cumfreq, buckets):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        #for i in range(0, self.numbuckets):
        for i in range(len(buckets)):
            if x >= buckets[i]['low'] and x < buckets[i]['high']:
                approx = None
                percentage = (x - buckets[i]['low']) / buckets[i]['size']
                if i > 0:
                    approx = percentage + cumfreq[i - 1]
                else:
                    approx = percentage * cumfreq[i]
                return approx / cumfreq[len(cumfreq) - 1]        

    def compute_histogram(self, N, sample):
        c = Counter(sample)
        sortedsample = sorted(list(set(sample)), key=float)
        low = sortedsample[0]
        high = sortedsample[1]
        leftedge = False
        for i in range(len(sortedsample)):
            # we know that the number of unique values in sorted sample is the same as the number of buckets we want
            # need to take care of edge cases when singular buckets could be those on the end
            if c[low] > N / self.numbuckets:
                # if the frequency of this value is greater than the constraint on regular buckets, then make it a
                # singular bucket
                b = {
                    'low': low,
                    'high': high,
                    'frequency': c[low],
                    'size': high - low
                }
                self.singular.append(b)
                #singlow = high
                if len(self.regular) != 0:
                    self.regular[len(self.regular) - 1]['high'] = high
                else:
                    # this is an edge case when the singular bucket is on the left edge and at the end, we have to go
                    # go back and extend the left bucket of the regular buckets if it exists
                    leftedge = True
            else:
                # otherwise we can just add the bucket as we normally would
                b = {
                    'low': low,
                    'high': high,
                    'frequency': c[low],
                    'size': high - low
                }
                self.regular.append(b)
            low = high
            if i < len(sortedsample) - 2:
                high = sortedsample[i + 2]
            else:
                high = low + 1
        if leftedge:
            # don't see how we could not end up having any regular buckets, so we always have at least one regular
            self.regular[0]['low'] = self.min
            self.regular[0]['size'] = self.regular[0]['high'] - self.regular[0]['low']

        assert len(self.singular) + len(self.regular) == self.numbuckets


    def add_datapoint(self, value, N, sample, attr):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value == self.min:
            # need to find bucket at end of range
            if len(self.singular) != 0 and self.singular[0]['low'] == self.regular[0]['low']:
                # then there is a singular bucket at the left extreme of the range in which case we extend
                # the ranges of both buckets while only adding to the frequency of the singular bucket
                self.singular[0]['low'] = self.min
                self.singular[0]['frequency'] += 1
                self.singular[0]['size'] = self.singular[0]['high'] - self.singular[0]['low']
                self.regular[0]['low'] = self.min
                self.regular[0]['size'] = self.regular[0]['high'] - self.regular[0]['low']
            else:
                # then there isn't a singular bucket at the left extreme of the range and we only extend the leftmost
                # regular bucket
                self.regular[0]['low'] = self.min
                self.regular[0]['frequency'] += 1
                self.regular[0]['size'] = self.regular[0]['high'] - self.regular[0]['low']

        elif value == self.max:
            if len(self.singular) != 0 and self.singular[len(self.singular) - 1]['high'] == self.regular[len(self.regular) - 1]['high']:
                # then there is a singular bucket at the right extreme of the range in which case we extend
                # the ranges of both buckets while only adding to the frequency of the singular bucket
                self.singular[len(self.singular) - 1]['high'] = self.max
                self.singular[len(self.singular) - 1]['frequency'] += 1
                self.singular[len(self.singular) - 1]['size'] = self.singular[len(self.singular) - 1]['high'] - self.singular[len(self.singular) - 1]['low']
                self.regular[len(self.regular) - 1]['high'] = self.max
                self.regular[len(self.regular) - 1]['size'] = self.regular[len(self.regular) - 1]['high'] - self.regular[len(self.regular) -1]['low']
            else:
                # then there isn't a singular bucket at the right extreme of the range and we only extend the rightmost
                # regular bucket
                self.regular[len(self.regular) - 1]['high'] = self.max
                self.regular[len(self.regular) - 1]['frequency'] += 1
                self.regular[len(self.regular) - 1]['size'] = self.regular[len(self.regular) - 1]['high'] - self.regular[len(self.regular) - 1]['low']

        else:
            self.checkbucketsandincrement(value, N)

        if self.chisquaretest() < 0.05:
            self.significanceReached(N)

    def demotebucket(self, bucket):
        # this method demotes a singular bucket to regular buckets
        low = bucket['low']
        high = bucket['high']
        for i in range(len(self.regular)):
            if bucket['size'] == 0:
                break
            elif self.regular[i]['low'] <= bucket['low'] < self.regular[i]['high'] and self.regular[i]['low'] < bucket['high'] <= self.regular[i]['high']:
                self.regular[i]['frequency'] += bucket['frequency']
                bucket['size'] = 0
                # if the bucket is completely contained within another regular bucket, then split that bucket evenly
                # might need to come up with something different because of the last condition
            elif self.regular[i]['low'] <= bucket['low'] < self.regular[i]['high'] and bucket['high'] > self.regular[i]['high']:
                # the case when the left boundary overlaps with a bucket but only part of the bucket will be added to this bucket
                frac = (self.regular[i]['high'] - bucket['low']) / bucket['size']
                perc = frac * bucket['frequency']
                self.regular[i]['frequency'] += perc
                bucket['frequency'] -= perc
                bucket['low'] = self.regular[i]['high']
                bucket['size'] = bucket['high'] - bucket['low']
        # now we need to redistribute the bucket boundaries so that the bucket counts remain the same
        # the question remains on how exactly to do this, simply add another boundary for another bucket?
        frequency = 0
        for i in range(len(self.regular)):
            frequency += self.regular[i]['frequency']
        threshold = frequency / (len(self.regular) + 1)
        self.redistributebuckets(threshold, True)
        for i in range(len(self.singular)):
            if self.singular[i]['low'] == low and self.singular[i]['high'] == high:
                del self.singular[i]
                break


    def promotebucket(self, bucket, N):
        # something we need to be aware of is the possibility of adding more than one bucket to the singular buckets, in
        # which case we need to rearrange the bucket boundaries in an equi-width manner
        low = bucket['low']
        high = bucket['high']
        singlength = len(self.singular)
        if len(self.singular) == 0:
            # if there are no buckets in self.singular then just append the bucket to the empty list
            self.singular.append(bucket)
        else:
            #index = []
            #singlen = len(self.singular)
            if bucket['low'] < self.singular[0]['low'] and self.singular[0]['low'] <= bucket['high']:
                frac = (self.singular[0]['low'] - bucket['low']) / bucket['size']
                perc = frac * bucket['frequency']
                b = {
                    'low': bucket['low'],
                    'high': self.singular[0]['low'],
                    'frequency': perc,
                    'size': self.singular[0]['low'] - bucket['low']
                }
                bucket['frequency'] -= perc
                bucket['low'] = self.singular[0]['low']
                bucket['size'] = bucket['high'] - bucket['low']
                self.singular.insert(0, b)
            elif bucket['low'] < self.singular[0]['low'] and bucket['high'] < self.singular[0]['low']:
                self.singular.insert(0, bucket.copy())
                bucket['size'] = 0
            i = 0
            #for i in range(len(self.singular)):
            while i < len(self.singular):
                if bucket['size'] == 0:
                    break
                elif self.singular[i]['low'] <= bucket['low'] < self.singular[i]['high'] and self.singular[i]['low'] < bucket['high'] <= self.singular[i]['high']:
                    # "SPECIAL CASE" when the bucket to insert fits within one bucket that is already present
                    # we need to increment bucket frequency and split the bucket
                    #index.append(i)
                    self.singular[i]['frequency'] += bucket['frequency']
                    bucket['size'] = 0
                    b = {
                        'low': self.singular[i]['low'] + (self.singular[i]['size'] / 2),
                        'high': self.singular[i]['high'],
                        'frequency': self.singular[i]['frequency'] / 2,
                        'size': self.singular[i]['high'] - (self.singular[i]['low'] + (self.singular[i]['size'] / 2))
                    }
                    self.singular[i]['frequency'] = self.singular[i]['frequency'] / 2
                    self.singular[i]['high'] = self.singular[i]['low'] + (self.singular[i]['size'] / 2)
                    self.singular[i]['size'] = self.singular[i]['size'] / 2
                    self.singular.insert(i + 1, b.copy())
                    break
                elif self.singular[i]['low'] <= bucket['low'] < self.singular[i]['high'] and bucket['high'] > self.singular[i]['high']:
                    # this is the case when the left boundary of the bucket overlaps with a bucket in the singular
                    fraction = (self.singular[i]['high'] - bucket['low']) / bucket['size']
                    perc = fraction * bucket['frequency']
                    self.singular[i]['frequency'] += perc
                    bucket['frequency'] -= perc
                    bucket['low'] = self.singular[i]['high']
                    bucket['size'] = bucket['high'] - bucket['low']
                    i -= 1
                elif bucket['low'] >= self.singular[i]['high']:
                    if i < len(self.singular) - 1 and bucket['low'] < self.singular[i + 1]['low']:
                        if self.singular[i]['high'] < bucket['high'] <= self.singular[i + 1]['low']:
                            # then there is a bucket that can fit in the gap between two buckets if the two buckets are not directly next to each other
                            frac = (bucket['high'] - bucket['low']) / bucket['size']
                            perc = frac * bucket['frequency']
                            b = {
                                'low': bucket['low'],
                                'high': bucket['high'],
                                'frequency': perc,
                                'size': bucket['high'] - bucket['low']
                            }
                            bucket['frequency'] -= perc
                            bucket['low'] = bucket['high']
                            bucket['size'] = bucket['high'] - bucket['low']
                            self.singular.insert(i + 1, b)
                        elif self.singular[i + 1]['low'] < bucket['high']:
                            # then there is the bucket to promote spills over into another bucket
                            frac = (self.singular[i + 1]['low'] - bucket['low']) / bucket['size']
                            perc = frac * bucket['frequency']
                            b = {
                                'low': bucket['low'],
                                'high': self.singular[i + 1]['low'],
                                'frequency': perc,
                                'size': self.singular[i + 1]['low'] - bucket['low']
                            }
                            # self.regular[i]['low'] += perc
                            bucket['frequency'] -= perc
                            bucket['low'] = self.singular[i + 1]['low']
                            bucket['size'] = bucket['high'] - bucket['low']
                            self.singular.insert(i + 1, b.copy())
                            i -= 1
                    elif i >= len(self.singular) - 1:
                        # then this bucket spills over into the end range of the singular buckets and we can just append the bucket
                        self.singular.append(bucket)
                        break
                i += 1
        # when we remove bucket we must extend the boundaries of the regular buckets
        for i in range(len(self.regular)):
            if self.regular[i]['low'] == low and self.regular[i]['high'] == high:
                del self.regular[i]
                break

        f = 0
        for j in range(len(self.regular)):
            f += self.regular[j]['frequency']
        for j in range(len(self.singular)):
            f += self.singular[j]['frequency']
        print f,N
        assert np.isclose(f, N)

        if len(self.singular) > singlength + 1:
            # then we have added more than 1 extra bucket and we need to redistribute the buckets within that range
            extra = len(self.singular) - (singlength + 1)
            self.redistributesingbuckets(low, high, extra, N)

        f = 0
        for j in range(len(self.regular)):
            f += self.regular[j]['frequency']
        for j in range(len(self.singular)):
            f += self.singular[j]['frequency']
        assert np.isclose(f, N)

    def redistributesingbuckets(self, low, high, extra, N):
        lowindex = None # low index
        highindex = None # high index
        for i in range(len(self.singular)):
            if self.singular[i]['low'] <= low < self.singular[i]['high']:
                lowindex = i
            elif self.singular[i]['low'] < high <= self.singular[i]['high']:
                highindex = i
                break
        f = 0
        for j in range(len(self.regular)):
            f += self.regular[j]['frequency']
        for j in range(len(self.singular)):
            f += self.singular[j]['frequency']
        assert np.isclose(f, N)
        freq = 0
        numbuckets = 0
        for i in range(lowindex, highindex + 1):
            freq += self.singular[i]['frequency']
            numbuckets += 1
        numbuckets -= extra
        eq = freq / numbuckets
        singular = []
        l = self.singular[lowindex]['low']
        h = None
        frequency = 0.0
        totalfreq = 0.0
        for i in range(lowindex, highindex + 1):
            if np.isclose(frequency,eq):
                h = self.singular[i]['low']
                b = {
                    'low': l,
                    'high': h,
                    'size': h - l,
                    'frequency': eq
                }
                totalfreq += eq
                frequency = 0.0
                l = h
                h = None
                singular.append(b.copy())
            elif frequency + self.singular[i]['frequency'] < eq:
                frequency += self.singular[i]['frequency']
            elif np.isclose(frequency + self.singular[i]['frequency'], eq):
                h = self.singular[i]['high']
                b = {
                    'low': l,
                    'high': h,
                    'frequency': eq,
                    'size': h - l
                }
                singular.append(b.copy())
                totalfreq += eq
                l = h
                h = None
                frequency = 0
            else:
                # the frequency of the bucket is too big for the threshold so we must recalculate the boundary
                # an edge case is that this bucket can be split into more than two buckets
                #l = self.singular[i]['low']
                while frequency + self.singular[i]['frequency'] > eq:
                    # how much do we need to take from this bucket?
                    freqperc = (eq - frequency) / self.singular[i]['frequency']
                    h = (freqperc * self.singular[i]['size']) + self.singular[i]['low']
                    b = {
                        'low': l,
                        'high': h,
                        'frequency': eq,
                        'size': h - l
                    }
                    totalfreq += eq
                    singular.append(b.copy())
                    l = h
                    h = None
                    frequency = 0.0
                    self.singular[i]['frequency'] -= (freqperc * self.singular[i]['frequency'])
                    self.singular[i]['low'] = l
                    self.singular[i]['size'] = self.singular[i]['high'] - self.singular[i]['low']
                frequency = self.singular[i]['frequency']
        if len(singular) < numbuckets:
            # then there we simply need to add the last bucket
            leftover = freq - totalfreq
            b = {
                'low': l,
                'high': self.singular[highindex]['high'],
                'frequency': leftover,
                'size': self.singular[highindex]['high'] - l
            }
            totalfreq += leftover
            singular.append(b.copy())
        assert np.isclose(totalfreq, freq)
        j = lowindex
        for i in range(lowindex, highindex + 1):
            del self.singular[lowindex]
            #if j < len(singular):
            #    self.singular[i] = singular[j]
            #else:
            #    del self.singular[i]
            #j += 1
        for i in range(len(singular)):
            self.singular.insert(j, singular[i])
            j += 1
        assert len(self.singular) + len(self.regular) == self.numbuckets
        f = 0
        for j in range(len(self.regular)):
            f += self.regular[j]['frequency']
        for j in range(len(self.singular)):
            f += self.singular[j]['frequency']
        assert np.isclose(f, N)

    def checkbucketsandincrement(self, value, N):
        # this method first checks the singular buckets and then the regular buckets, and promotes regular buckets if larger than threshold
        for i in range(len(self.singular)):
            if self.singular[i]['low'] <= value < self.singular[i]['high']:
                self.singular[i]['frequency'] += 1
                return
        for i in range(len(self.regular)):
            if self.regular[i]['low'] <= value < self.regular[i]['high']:
                self.regular[i]['frequency'] += 1
                return

    def chisquaretest(self):
        observed = []
        reg = 0
        freq = 0
        for bucket in self.regular:
            reg += 1
            freq += bucket['frequency']
            observed.append(bucket['frequency'])
        avg = freq / reg
        expected = np.array([0] * len(observed))
        expected.fill(avg)
        observed = np.array(observed)
        chisquare = stats.chisquare(f_obs=observed, f_exp=expected)
        return chisquare[1]

    def redistributebuckets(self, threshold, add):
        reglen = len(self.regular)
        regular = []
        low = self.min
        high = None
        frequency = 0.0
        for i in range(len(self.regular)):
            if np.isclose(frequency, threshold):
                high = self.regular[i]['low']
                b = {
                    'low': low,
                    'high': high,
                    'size': high - low,
                    'frequency': threshold
                }
                frequency = 0.0
                low = high
                high = None
                regular.append(b.copy())
            if frequency + self.regular[i]['frequency'] < threshold:
                frequency += self.regular[i]['frequency']
            elif np.isclose(frequency + self.regular[i]['frequency'], threshold):
                high = self.regular[i]['high']
                b = {
                    'low': low,
                    'high': high,
                    'frequency': threshold,
                    'size': high - low
                }
                regular.append(b.copy())
                low = high
                high = None
                frequency = 0
            else:
                # the frequency of the bucket is too big for the threshold so we must recalculate the boundary
                # an edge case is that this bucket can be split into more than two buckets
                while frequency + self.regular[i]['frequency'] > threshold:
                    # how much do we need to take from this bucket?
                    freqperc = (threshold - frequency) / self.regular[i]['frequency']
                    high = (freqperc * self.regular[i]['size']) + self.regular[i]['low']
                    b = {
                        'low': low,
                        'high': high,
                        'frequency': threshold,
                        'size': high - low
                    }
                    regular.append(b.copy())
                    low = high
                    high = None
                    frequency = 0
                    self.regular[i]['frequency'] -= freqperc * self.regular[i]['frequency']
                    self.regular[i]['low'] = low
                    self.regular[i]['size'] = self.regular[i]['high'] - self.regular[i]['low']
                frequency = self.regular[i]['frequency']
        if len(regular) < reglen or (add and len(regular) == reglen):
            # then there we simply need to add the last bucket
            b = {
                'low': low,
                'high': self.max + 1,
                'frequency': threshold,
                'size': self.max + 1 - low
            }
            regular.append(b.copy())
        self.regular = regular

        if add:
            print len(self.regular), reglen
            assert len(self.regular) == reglen + 1
        else:
            assert len(self.regular) == reglen

    def significanceReached(self, N):
        print "signficance reached"
        count = N / self.numbuckets
        s = 0
        i = 0
        while i < len(self.regular):
            if round(self.regular[i]['frequency'], 10) > count:
                f = 0
                for j in range(len(self.regular)):
                    f += self.regular[j]['frequency']
                for j in range(len(self.singular)):
                    f += self.singular[j]['frequency']
                assert np.isclose(f, N)
                self.promotebucket(self.regular[i].copy(), N)
                assert len(self.regular) + len(self.singular) == self.numbuckets
                f = 0
                for j in range(len(self.regular)):
                    f += self.regular[j]['frequency']
                for j in range(len(self.singular)):
                    f += self.singular[j]['frequency']
                assert np.isclose(f, N)
                i = 0
            else:
                i += 1

        if self.regular[0]['low'] != self.min:
            self.regular[0]['low'] = self.min
            self.regular[0]['size'] = self.regular[0]['high'] - self.regular[0]['low']

        if self.regular[len(self.regular) - 1]['high'] != self.max + 1:
            self.regular[len(self.regular) - 1]['high'] = self.max + 1
            self.regular[len(self.regular) - 1]['size'] = self.regular[len(self.regular) - 1]['high'] - self.regular[len(self.regular) - 1]['low']

        for i in range(len(self.regular)):
            s += self.regular[i]['frequency']
            h = self.regular[i]['high']
            if i < len(self.regular) - 1 and h != self.regular[i + 1]['low']:
                rang = self.regular[i + 1]['low'] - h
                self.regular[i]['high'] += rang / 2
                self.regular[i]['size'] = self.regular[i]['high'] - self.regular[i]['low']
                self.regular[i + 1]['low'] -= rang / 2
                self.regular[i + 1]['size'] = self.regular[i + 1]['high'] - self.regular[i + 1]['low']

        threshold = s / len(self.regular)
        # now we need to redistribute the regular buckets and making sure that each bucket has the threshold frequency
        f = 0
        for i in range(len(self.regular)):
            f += self.regular[i]['frequency']
        for i in range(len(self.singular)):
            f += self.singular[i]['frequency']
        assert np.isclose(f, N)
        self.redistributebuckets(threshold, False)
        f = 0
        for i in range(len(self.regular)):
            f += self.regular[i]['frequency']
        for i in range(len(self.singular)):
            f += self.singular[i]['frequency']
        assert np.isclose(f, N)
        # if there are regular buckets whose frequency exceeds count, make that bucket a non-regular bucket
        assert len(self.regular) + len(self.singular) == self.numbuckets
        i = 0
        while i < len(self.singular):
            if round(self.singular[i]['frequency'], 10) < count:
                f = 0
                for j in range(len(self.regular)):
                    f += self.regular[j]['frequency']
                for j in range(len(self.singular)):
                    f += self.singular[j]['frequency']
                assert np.isclose(f, N)
                #print self.singular[i]['low'], self.singular[i]['high']
                self.demotebucket(self.singular[i].copy())
                assert len(self.regular) + len(self.singular) == self.numbuckets
                f = 0
                for j in range(len(self.regular)):
                    f += self.regular[j]['frequency']
                for j in range(len(self.singular)):
                    f += self.singular[j]['frequency']
                #print f, N
                assert np.isclose(f, N)
                i = 0
            else:
                i += 1
        f = 0
        for i in range(len(self.regular)):
            f += self.regular[i]['frequency']
        for i in range(len(self.singular)):
            f += self.singular[i]['frequency']
        assert np.isclose(f, N)

    def plot_histogram(self, attr, buckets):
        """Plots the histogram."""
        #
        # totalbuckets = copy.deepcopy(self.regular)
        # for i in range(len(self.singular)):
        #     bucket = self.singular[i].copy()
        #     for j in range(len(totalbuckets)):
        #         if bucket['size'] == 0:
        #             break
        #         elif totalbuckets[i]['low'] <= bucket['low'] < totalbuckets[i]['high'] and totalbuckets[i]['low'] < \
        #                 bucket['high'] <= totalbuckets[i]['high']:
        #             totalbuckets[i]['frequency'] += bucket['frequency']
        #             bucket['size'] = 0
        #             # if the bucket is completely contained within another regular bucket, then split that bucket evenly
        #             # might need to come up with something different because of the last condition
        #         elif totalbuckets[i]['low'] <= bucket['low'] < totalbuckets[i]['high'] and bucket['high'] > \
        #                 totalbuckets[i]['high']:
        #             # the case when the left boundary overlaps with a bucket but only part of the bucket will be added to this bucket
        #             frac = (totalbuckets[i]['high'] - bucket['low']) / bucket['size']
        #             perc = frac * bucket['frequency']
        #             totalbuckets[i]['frequency'] += perc
        #             bucket['frequency'] -= perc
        #             bucket['low'] = totalbuckets[i]['high']
        #             bucket['size'] = bucket['high'] - bucket['low']



        bins = []
        frequency = []
        for b in buckets:
            bins.append(b['low'])
            frequency.append(b['frequency'])
        bins.append(b['high'])

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
        plt.close()
        self.counter += 1

    def print_buckets(self, buckets):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        high = buckets[0]['low']
        for i in range(0, len(buckets)):
            print "---------------- bucket " + str(i) + " ----------------"
            for k, v in buckets[i].iteritems():
                print str(k) + ": " + str(v)
            print "------------------- END -------------------"
            high = buckets[i]['high']

    def checkfrequency(self):
        # checks to make sure frequency is a whole number, returns True if it's not
        f = 0.0
        for i in range(len(self.singular)):
            f += self.singular[i]['frequency']
        for i in range(len(self.regular)):
            f = f + self.regular[i]['frequency']
        check = f % 1.0
        if check == 1.0:
            return True
        else:
            return False
