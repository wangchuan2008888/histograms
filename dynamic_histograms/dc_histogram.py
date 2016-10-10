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
        #buckets = []
        # for i in range(0, self.numbuckets):
        #     buckets.append({
        #         'low': 0,
        #         'high': 0,
        #         'frequency': 0,
        #         'size': 0,
        #         'regular': True
        #     })
        # self.buckets = buckets
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

    def create_histogram(self, attr, gamma, gammam, batchsize, userbucketsize):
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
                if float(row[attr_index]) < self.min:
                    self.min = float(row[attr_index])
                if float(row[attr_index]) > self.max:
                    self.max = float(row[attr_index]) 
                if len(set(sample)) < self.numbuckets:
                    sample.append(float(row[attr_index]))
                elif len(set(sample)) == self.numbuckets and initial == False:
                    self.compute_histogram(N, sample)
                    self.plot_histogram(attr, self.regular + self.singular)
                    #d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    #d.create_distribution(self.regular)
                    #new_buckets = d.return_distribution()
                    #self.plot_histogram(attr, new_buckets)
                    #skip = self.calculateSkip(len(sample))
                    initial = True
                elif initial == True:
                    #skipcounter += 1
                    self.add_datapoint(float(row[attr_index]), N, sample, attr, gamma, gammam)
                    #if skipcounter == skip:
                    #    sample = self.maintainBackingSample(float(row[attr_index]), sample)
                    #    skip = self.calculateSkip(len(sample))
                    #    skipcounter = 0
                    if N % batchsize == 0:
                        print "number read in: " + str(N)
                        self.plot_histogram(attr, self.regular)
                        #d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        #d.create_distribution(self.regular)
                        #new_buckets = d.return_distribution()
                        #self.plot_histogram(attr, new_buckets)
                        #self.compare_histogram(attr, False)
                        f = 0
                        for i in range(len(self.regular)):
                            f += self.regular[i]['frequency']
                        for i in range(len(self.singular)):
                            f += self.singular[i]['frequency']
                        print f, N
                        #print N
                        #assert f == N
                else:
                    print("ERROR: There are not enough unique values for the number of specified buckets.")
                N += 1
        #self.compare_histogram(attr, False)

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
        # if len(self.singular) != 0:
        #     print('SINGULAR BUCKETS')
        #     self.print_buckets(self.singular)
        # print('REGULAR BUCKETS')
        # self.print_buckets(self.regular)
        assert len(self.singular) + len(self.regular) == self.numbuckets
        if self.checkfrequency():
            print "FREQUENCY IS FUCKED UP computing"
            self.print_buckets(self.regular)
            if len(self.singular) != 0:
                print "SINGULAR"
                self.print_buckets(self.singular)
            sys.exit()
        # l = N / len(sample)
        # betaprime = self.numbuckets
        # mprime = len(sample)
        # c = Counter(sample)
        # mostfreq = c.most_common(self.numbuckets + 1)
        # mostfreq = sorted(mostfreq, key=lambda x: x[0], reverse=True)
        # low2 = min(sample)
        # buckets = self.buckets
        # high = mostfreq[0][0] + 1
        # low = mostfreq[0][0]
        # for i in range(0, self.numbuckets - 1):
        #     if c[mostfreq[i][0]] >= mprime / betaprime:
        #         buckets[betaprime - 1]['high'] = high
        #         buckets[betaprime - 1]['low'] = low
        #         buckets[betaprime - 1]['frequency'] = l * c[mostfreq[i][0]]
        #         buckets[betaprime - 1]['regular'] = False
        #         buckets[betaprime - 1]['size'] = high - low
        #         mprime -= c[mostfreq[i][0]]
        #         betaprime -= 1
        #         high = low
        #         low = mostfreq[i + 1][0]
        # sample = sorted(sample)
        # for i in range(1, betaprime):
        #     buckets[i - 1]['high'] = sample[i * (mprime // betaprime)]
        #     buckets[i - 1]['frequency'] = l * (mprime / betaprime)
        #     buckets[i - 1]['size'] = buckets[i - 1]['high'] - buckets[i - 1]['low']
        # for i in range(0, len(buckets)):
        #     buckets[i]['low'] = low2
        #     buckets[i]['size'] = buckets[i]['high'] - buckets[i]['low']
        #     low2 = buckets[i]['high']
        # self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        # self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        # self.buckets[0]['low'] = self.min
        # self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        # self.split = (2 + gamma) * (l * mprime / betaprime)
        # self.merge = (l * mprime) / ((2 + gammam) * betaprime)
        # #print sample
        # #self.print_buckets()
        # #print self.min,self.max
        # self.buckets = buckets

    # def calculateSkip(self, n):
    #     v = random.uniform(0, 1)
    #     l = 0
    #     t = n + 1
    #     num = 1
    #     quot = num / t
    #     while quot > v:
    #         l += 1
    #         t += 1
    #         num += 1
    #         quot = (quot * num) / t
    #     return l
    #
    # def maintainBackingSample(self, value, sample):
    #     if len(sample) + 1 <= self.upper:
    #         sample.append(value)
    #     else:
    #         rand_index = random.randint(0,len(sample) - 1)
    #         sample[rand_index] = value
    #     return sample

    # def mergebucketlists(self):
    #     if len(self.singular) == 0:
    #         return self.regular
    #     else:
    #         buckets = self.singular
    #         regular = True
    #         if self.singular[0]['low'] == self.min:
    #             regular = False
    #         if regular:
    #             regularlen = len(self.regular)
    #             for i in range(regularlen):
    #                 for j in range(len(buckets)):
    #                     if


    def add_datapoint(self, value, N, sample, attr, gamma, gammam):
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
                #if self.regular[0]['frequency'] > N / self.numbuckets:
                #    self.promotebucket(self.regular[0])
            # self.buckets[0]['low'] = value
            # self.buckets[0]['frequency'] += 1
            # self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
            # if self.buckets[0]['frequency'] > N / self.numbuckets:
            #     self.buckets[0]['regular'] = False
            #if self.buckets[0]['frequency'] >= self.split and self.buckets[0]['regular'] == True:
            #    self.splitbucket(N, 0, None, 1, sample, gamma, gammam)
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
                #if self.regular[len(self.regular) - 1]['frequency'] > N / self.numbuckets:
                #    self.promotebucket(self.regular[len(self.regular) - 1])
            # self.buckets[self.numbuckets - 1]['high'] = value + 1
            # self.buckets[self.numbuckets - 1]['frequency'] += 1
            # self.buckets[self.numbuckets - 1]['size'] = value + 1 - self.buckets[self.numbuckets - 1]['low']
            # if self.buckets[self.numbuckets - 1]['frequency'] > N / self.numbuckets:
            #     self.buckets[self.numbuckets - 1]['regular'] = False
            #if self.buckets[self.numbuckets - 1]['frequency'] >= self.split and self.buckets[self.numbuckets - 1]['regular'] == True:
            #    self.splitbucket(N, self.numbuckets - 1, self.numbuckets - 2, None, sample, gamma, gammam)
        else:
            self.checkbucketsandincrement(value, N)
            # for i in range(0, self.numbuckets):
            #     if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
            #         self.buckets[i]['frequency'] += 1
            #         if self.buckets[i]['frequency'] > N / self.numbuckets:
            #             self.buckets[i]['regular'] = False
                    #if self.buckets[i]['frequency'] >= self.split and self.buckets[i]['regular'] == True:
                    #    if i == 0:
                    #        self.splitbucket(N, 0, None, 1, sample, gamma, gammam)
                    #    elif i == self.numbuckets - 1:
                    #        self.splitbucket(N, i, i - 1, None, sample, gamma, gammam)
                    #    else:
                    #        self.splitbucket(N, i, i - 1, i + 1, sample, gamma, gammam)
        if self.chisquaretest() < 0.05:
            self.significanceReached(N)

    def demotebucket(self, bucket):
        # this method demotes a singular bucket to regular buckets
        for i in range(len(self.regular)):
            #print i, len(self.regular)
            if bucket['size'] == 0:
                break
            elif self.regular[i]['low'] <= bucket['low'] < self.regular[i]['high'] and self.regular[i]['low'] < bucket['high'] <= self.regular[i]['high']:
                self.regular[i]['frequency'] += bucket['frequency']
                bucket['size'] = 0
                # if the bucket is completely contained within another regular bucket, then split that bucket evenly
                # might need to come up with something different because of the last condition
                # b = {
                #     'low': self.regular[i]['low'] + (self.regular[i]['size'] / 2),
                #     'high': self.regular[i]['high'],
                #     'frequency': self.regular[i]['frequency'] / 2,
                #     'size': self.regular[i]['high'] - (self.regular[i]['low'] + (self.regular[i]['size'] / 2))
                # }
                # self.regular[i]['high'] = b['low']
                # self.regular[i]['frequency'] = self.regular[i]['frequency'] / 2
                # self.regular[i]['size'] = self.regular[i]['high'] - self.regular[i]['low']
                # self.regular.insert(i + 1, b)
            elif self.regular[i]['low'] <= bucket['low'] < self.regular[i]['high'] and bucket['high'] > self.regular[i]['high']:
                # the case when the left boundary overlaps with a bucket but only part of the bucket will be added to this bucket
                frac = (self.regular[i]['high'] - bucket['low']) / bucket['size']
                perc = frac * bucket['frequency']
                self.regular[i]['low'] += perc
                bucket['frequency'] -= perc
                bucket['low'] = self.regular[i]['high']
                bucket['size'] = bucket['high'] - bucket['low']
            #elif i < len(self.singular) - 1 and bucket['low'] > self.singular[i]['high'] and self.regular[i + 1]['low'] < bucket['high'] <= self.regular[i + 1]['high']:
                # then this bucket to add extends past this bucket and into the
        # now we need to redistribute the bucket boundaries so that the bucket counts remain the same
        # the question remains on how exactly to do this, simply add another boundary for another bucket?
        frequency = 0
        for i in range(len(self.regular)):
            frequency += self.regular[i]['frequency']
        threshold = frequency / (len(self.regular) + 1)
        self.redistributebuckets(threshold, True)
        self.singular.remove(bucket)


    def promotebucket(self, bucket):
        # something we need to be aware of is the possibility of adding more than one bucket to the singular buckets, in
        # which case we need to rearrange the bucket boundaries in an equi-width manner
        if len(self.singular) == 0:
            # if there are no buckets in self.singular then just append the bucket to the empty list
            self.singular.append(bucket)
        else:
            #index = []
            #singlen = len(self.singular)
            for i in range(len(self.singular)):
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
                        'size': self.singular[i]['high'] - self.singular[i]['low'] + (self.singular[i]['size'] / 2)
                    }
                    self.singular[i]['frequency'] = self.singular[i]['frequency'] / 2
                    self.singular[i]['high'] = self.singular[i]['low'] + (self.singular[i]['size'] / 2)
                    self.singular[i]['size'] = self.singular[i]['size'] / 2
                    self.singular.insert(i + 1, b)
                    break
                elif self.singular[i]['low'] <= bucket['low'] < self.singular[i]['high'] and bucket['high'] > self.singular[i]['high']:
                    # this is the case when the left boundary of the bucket overlaps with a bucket in the singular
                    frac = (self.singular[i]['high'] - bucket['low']) / bucket['size']
                    perc = frac * bucket['frequency']
                    self.singular[i]['low'] += perc
                    bucket['frequency'] -= perc
                    bucket['low'] = self.singular[i]['high']
                    bucket['size'] = bucket['high'] - bucket['low']
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
                            self.singular.insert(i + 1, b)
                    elif i >= len(self.singular) - 1:
                        # then this bucket spills over into the end range of the singular buckets and we can just append the bucket
                        self.singular.append(bucket)
        # when we remove bucket we must extend the boundaries of the regular buckets
        for i in range(len(self.regular)):
            if self.regular[i]['low'] == bucket['low'] and self.regular[i]['high'] == bucket['high']:
                if i == len(self.regular) - 1:
                    # then we just extend the bucket to the left of it all the way
                    self.regular[i - 1]['high'] = bucket['high']
                    self.regular[i - 1]['size'] = self.regular[i - 1]['high'] - self.regular[i - 1]['low']
                elif i == 0:
                    # then we extend the bucket to the right of it all the way
                    self.regular[1]['low'] = bucket['low']
                    self.regular[1]['size'] = self.regular[1]['high'] - self.regular[1]['low']
                else:
                    # then we extend the left and right buckets only halfway
                    self.regular[i - 1]['high'] = bucket['low'] + (bucket['size'] / 2)
                    self.regular[i - 1]['size'] = self.regular[i - 1]['high'] - self.regular[i - 1]['low']
                    self.regular[i + 1]['low'] = self.regular[i - 1]['high']
                    self.regular[i + 1]['size'] = self.regular[i + 1]['high'] - self.regular[i + 1]['low']
                del self.regular[i]
                break

    def checkbucketsonright(self, i, bucket):
        if i != len(self.singular) - 1:
            # have to check if there are buckets on the right in the range of the bucket to be inserted
            # if not, then just insert what is left of the bucket next to the bucket it first overlapped with
            j = i + 1
            while bucket['frequency'] != 0 and self.singular[j]['high'] >= bucket['high'] and self.singular[j]['low'] < \
                    bucket['high'] and j < len(self.singular):
                # going until we find a bucket whose end matches the end of the bucket range to insert or is greater than it
                if self.singular[j - 1]['high'] == self.singular[j]['low']:
                    # then there is a bucket directly after the previous bucket and we must add and split if necessary
                    frac = (self.singular[j]['high'] - bucket['low']) / bucket['size']
                    perc = bucket['frequency'] * frac
                    self.singular[j]['frequency'] += perc
                    bucket['low'] = self.singular[j]['high']
                    bucket['frequency'] -= perc
                    bucket['size'] = bucket['high'] - bucket['low']
                    j += 1
                else:
                    # otherwise there is not a bucket directly after the bucket and we must then insert the bucket in
                    # the appropriate range, but cognizant of the fact that there is still a bucket that overlaps the range
                    rang = self.singular[j]['low'] - bucket['low']
                    perc = (rang / bucket['size']) * bucket['frequency']
                    b = {
                        'low': bucket['low'],
                        'high': self.singular[j]['low'],
                        'frequency': perc,
                        'size': rang
                    }
                    bucket['low'] = self.singular[j]['low']
                    bucket['frequency'] -= perc
                    bucket['size'] = bucket['high'] - bucket['low']
                    leftover = bucket['high'] - self.singular[j]['low']
                    # in the case that it spills over into another bucket but not completely
                    if leftover > self.singular[j]['size']:
                        # then it spills over into the entire bucket and then some
                        leftover = self.singular[j]['high'] - bucket['low']
                        bucket['low'] = self.singular[j]['high']
                    leftoverperc = (leftover / bucket['size']) * bucket['frequency']
                    self.singular[j]['frequency'] += leftoverperc
                    bucket['frequency'] -= leftoverperc
                    bucket['size'] = bucket['high'] - bucket['low']
                    self.singular.insert(j, b)
                    j += 2
            if bucket['frequency'] != 0:
                # then we add the bucket at self.singular[j] because there is no bucket in which to insert frequencies already
                self.singular.insert(j, bucket)

        else:
            # otherwise this is the last bucket in singular and we can just add the remaining bucket to the end of the singular list
            self.singular.append(bucket)

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
                #if self.regular[i]['frequency'] > N / self.numbuckets:
                #    self.promotebucket(self.regular[i])
                #    return

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
        frequency = 0
        for i in range(len(self.regular)):
            if frequency == threshold:
                high = self.regular[i]['low']
                b = {
                    'low': low,
                    'high': high,
                    'size': high - low,
                    'frequency': threshold
                }
                frequency = 0
                low = high
                high = None
                regular.append(b.copy())
            elif frequency + self.regular[i]['frequency'] < threshold:
                frequency += self.regular[i]['frequency']
            elif frequency + self.regular[i]['frequency'] == threshold:
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
                    frequency = 0  # (1 - freqperc) * self.regular[i]['frequency']
                    self.regular[i]['frequency'] -= freqperc * self.regular[i]['frequency']
                    self.regular[i]['low'] = low
                    self.regular[i]['size'] = self.regular[i]['high'] - self.regular[i]['low']
                frequency = self.regular[i]['frequency']
                #low = self.regular[i]['low']
                #high = None
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
        #print reglen, self.min, self.max
        ##print len(self.regular)
        #print add
        #print "DONE"
        #self.print_buckets(self.regular)
        if add:
            assert len(self.regular) == reglen + 1
        else:
            assert len(self.regular) == reglen

    def significanceReached(self, N):
        print "signficance reached"
        count = N / self.numbuckets
        s = 0
        #ranges = []
        # rangesum = 0
        # r = False
        # low = None
        # high = None
        #for i in range(len(self.singular)):
        i = 0
        while i < len(self.singular):
            if self.singular[i]['frequency'] < count:
                self.demotebucket(self.singular[i])
                i = 0
                print "DEMOTING DEMOTING"
            else:
                i += 1
        if self.checkfrequency():
            print "FREQUENCY IS FUCKED UP demote"
            self.print_buckets(self.regular)
            if len(self.singular) != 0:
                print "SINGULAR"
                self.print_buckets(self.singular)
        for i in range(len(self.regular)):
            s += self.regular[i]['frequency']
        threshold = s / len(self.regular)
        # now we need to redistribute the regular buckets and making sure that each bucket has the threshold frequency
        self.redistributebuckets(threshold, False)
        if self.checkfrequency():
            print "FREQUENCY IS FUCKED UP redistribute"
            self.print_buckets(self.regular)
            if len(self.singular) != 0:
                print "SINGULAR"
                self.print_buckets(self.singular)
        # for i in range(len(self.buckets)):
        #     if r == False and self.buckets[i]['regular'] == True:
        #         low = i
        #         r = True
        #         rangesum += self.buckets[i]['frequency']
        #     elif r == True and self.buckets[i]['regular'] == False:
        #         high = i
        #         self.splitbucketrange(low, high, threshold, rangesum)
        #         low = None
        #         high = None
        #         rangesum = 0
        #         r = False
        #     elif r == True and self.buckets[i]['regular'] == True:
        #         rangesum += self.buckets[i]['frequency']
        # if low != None:
        #     high = i
        #     self.splitbucketrange(low, high, threshold, self.buckets[len(self.buckets) - 1]['frequency'])
        print len(self.regular), len(self.singular)
        assert len(self.regular) + len(self.singular) == self.numbuckets

        # if there are regular buckets whose frequency exceeds count, make that bucket a non-regular bucket
        #for i in range(len(self.regular)):
        i = 0
        while i < len(self.regular):
            if self.regular[i]['frequency'] > count:
                self.promotebucket(self.regular[i])
                i = 0
            else:
                i += 1
        if self.checkfrequency():
            print "FREQUENCY IS FUCKED UP promote"
            self.print_buckets(self.regular)
            if len(self.singular) != 0:
                print "SINGULAR"
                self.print_buckets(self.singular)
        print len(self.regular), len(self.singular)
        assert len(self.regular) + len(self.singular) == self.numbuckets

    # def splitbucketrange(self, low, high, count, summation):
    #     buckets = []
    #     bucket = {
    #         'low': self.buckets[low]['low'],
    #         'high': None,
    #         'frequency': 0,
    #         'size': 0,
    #         'regular': True
    #     }
    #     freq = 0
    #     for i in range(low, high):
    #         if freq == count:
    #             # then we can end the previous bucket and begin accumulating frequencies for the next bucket
    #             bucket['high'] = self.buckets[i]['low']
    #             bucket['size'] = bucket['high'] - bucket['low']
    #             bucket['frequency'] = count
    #             buckets.append(bucket.copy())
    #             bucket['low'] = self.buckets[i]['low']
    #             bucket['high'] = None
    #             bucket['size'] = 0
    #             freq = 0
    #         elif freq + self.buckets[i]['frequency'] == count:
    #             # including this bucket we can end the previous bucket and begin a new with the new bucket
    #             freq += self.buckets[i]['frequency']
    #             bucket['high'] = self.buckets[i]['high']
    #             bucket['frequency'] = count
    #             bucket['size'] = bucket['high'] - bucket['low']
    #             buckets.append(bucket.copy())
    #             bucket['low'] = bucket['high']
    #             bucket['high'] = None
    #             bucket['size'] = 0
    #             freq = 0
    #         elif freq + self.buckets[i]['frequency'] > count:
    #             # then we need to take a percentage of this bucket and split it, possibly into more than 2 buckets
    #             diff = self.buckets[i]['frequency'] - (freq + self.buckets[i]['frequency'] - count)
    #             percentage = diff / self.buckets[i]['frequency']
    #             assert percentage <= 1
    #             freq += self.buckets[i]['frequency'] * percentage
    #             bucket['high'] = self.buckets[i]['low'] + (self.buckets[i]['size'] * percentage)
    #             bucket['size'] = bucket['high'] - bucket['low']
    #             bucket['frequency'] = count
    #             buckets.append(bucket.copy())
    #             bucket['low'] = bucket['high']
    #             bucket['high'] = None
    #             bucket['size'] = 0
    #             freq = self.buckets[i]['frequency'] * (1 - percentage)
    #             self.buckets[i]['frequency'] = freq
    #             while freq > count:
    #                 percentage = count / self.buckets[i]['frequency']
    #                 bucket['high'] = bucket['low'] + ((self.buckets[i]['high'] - bucket['low']) * percentage)
    #                 bucket['size'] = bucket['high'] - bucket['low']
    #                 bucket['frequency'] = count
    #                 buckets.append(bucket.copy())
    #                 bucket['low'] = bucket['high']
    #                 bucket['high'] = None
    #                 bucket['size'] = 0
    #                 freq = self.buckets[i]['frequency'] * (1 - percentage)#((self.buckets[i]['high'] - bucket['low']) / self.buckets[i]['size'])
    #                 self.buckets[i]['frequency'] = freq
    #         elif freq + self.buckets[i]['frequency'] < count:
    #             freq += self.buckets[i]['frequency']
    #
    #     if summation % count != 0:
    #         remainder = summation % count
    #         if bucket['high'] == None: # then there are left over frequencies that have not been put into a bucket
    #             bucket['high'] = self.buckets[high]['low']
    #             bucket['frequency'] = freq
    #             bucket['size'] = bucket['high'] - bucket['low']
    #             buckets.append(bucket.copy())
    #         else:
    #             buckets[len(buckets) - 1]['frequency'] += remainder
    #
    #
    #     for i in range(low, high):
    #         del self.buckets[low]
    #     for i in range(len(buckets) - 1, -1, -1):
    #         self.buckets.insert(low, buckets[i])



    # def splitbucket(self, N, bucketindex, prevbucketindex, afterbucketindex, sample, gamma, gammam):
    #     s = []
    #     for i in range(0, len(sample)):
    #         if sample[i] >= self.buckets[bucketindex]['low'] and sample[i] <= self.buckets[bucketindex]['high']:
    #             s.append(sample[i])
    #     m = np.median(s)
    #     if prevbucketindex != None and m != self.buckets[prevbucketindex]['high'] and m != self.buckets[bucketindex]['high']:
    #         mergepair_index = self.candidateMergePair()
    #         if mergepair_index != None:
    #             self.mergebuckets(mergepair_index) # merge the buckets into one bucket
    #             self.splitbucketintwo(bucketindex, sample) # split bucket
    #         else:
    #             self.compute_histogram(N, sample, gamma, gammam)
    #     elif prevbucketindex != None and m == self.buckets[prevbucketindex]['high']:
    #         c = Counter(sample)
    #         self.buckets[bucketindex]['frequency'] = self.buckets[prevbucketindex]['frequency'] + self.buckets[bucketindex]['frequency'] - (c[m] * N / len(sample))
    #         self.buckets[prevbucketindex]['high'] = m
    #         self.buckets[prevbucketindex]['size'] = m - self.buckets[prevbucketindex]['low']
    #         self.buckets[prevbucketindex]['frequency'] = c[m] * N / len(sample)
    #         if self.buckets[bucketindex]['frequency'] >= self.split:
    #             self.splitbucket(N, bucketindex, prevbucketindex, afterbucketindex, sample)
    #         elif self.buckets[bucketindex]['frequency'] <= self.merge:
    #             mergepair_index = self.candidateMergePair()
    #             split_index = self.candidatesplitbucket(gamma)
    #             if mergepair_index != None and split_index != None and split_index > 0:
    #                 if self.buckets[bucketindex]['high'] == self.buckets[mergepair_index]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index]['frequency']:
    #                     self.mergebuckets(mergepair_index)
    #                     after = None
    #                     if split_index < self.numbuckets - 1:
    #                         after = split_index + 1
    #                     self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
    #                 elif self.buckets[bucketindex]['high'] == self.buckets[mergepair_index + 1]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index + 1]['frequency']:
    #                     self.mergebuckets(mergepair_index)
    #                     after = None
    #                     if split_index < self.numbuckets - 1:
    #                         after = self.buckets[split_index + 1]
    #                     self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
    #             else:
    #                 self.compute_histogram(N, sample, gamma, gammam)
    #     elif m == self.buckets[bucketindex]['high'] and afterbucketindex != None:
    #         c = Counter(sample)
    #         self.buckets[bucketindex]['frequency'] = self.buckets[afterbucketindex]['frequency'] + self.buckets[bucketindex]['frequency'] - (c[m] * N / len(sample))
    #         self.buckets[afterbucketindex]['high'] = m
    #         self.buckets[afterbucketindex]['size'] = m - self.buckets[afterbucketindex]['low']
    #         self.buckets[afterbucketindex]['frequency'] = c[m] * N / len(sample)
    #         if self.buckets[afterbucketindex]['frequency'] <= self.split:
    #             after = None
    #             if afterbucketindex < self.numbuckets - 1:
    #                 after = afterbucketindex + 1
    #             self.splitbucket(N, afterbucketindex, bucketindex, after, sample)
    #         elif self.buckets[afterbucketindex]['frequency'] <= self.merge:
    #             mergepair_index = self.candidateMergePair()
    #             split_index = self.candidatesplitbucket(self, gamma)
    #             if mergepair_index != None and split_index != None and split_index > 0:
    #                 if self.buckets[afterbucketindex]['high'] == self.buckets[mergepair_index]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index]['frequency']:
    #                     self.mergebuckets(mergepair_index)
    #                     after = None
    #                     if split_index < self.numbuckets - 1:
    #                         after = split_index + 1
    #                     self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
    #                 elif self.buckets[bucketindex]['high'] == self.buckets[mergepair_index + 1]['high'] and self.buckets[bucketindex]['frequency'] == self.buckets[mergepair_index + 1]['frequency']:
    #                     self.mergebuckets(mergepair_index)
    #                     after = None
    #                     if split_index < self.numbuckets - 1:
    #                         after = split_index + 1
    #                     self.splitbucket(N, split_index, split_index - 1, after, sample, gamma)
    #             else:
    #                 self.compute_histogram(N, sample, gamma, gammam)
        

    # def splitbucketintwo(self, index, sample):
    #     """Splits a bucket in the list of buckets of the histogram."""
    #     s = []
    #     for i in range(0, len(sample)):
    #         if sample[i] >= bucket['low'] and sample[i] < bucket['high']:
    #             s.append(sample[i])
    #     m = np.median(s)
    #     s = list(set(s))
    #     bucket2 = {
    #         'low': s[s.index(m) + 1],
    #         'high': self.buckets[index]['high'],
    #         'size': self.buckets[index]['high'] - s[s.index(m) + 1],
    #         'frequency': self.split / 2
    #     }
    #     self.buckets[index]['high'] = s[s.index(m) + 1]
    #     self.buckets[index]['frequency'] = self.split / 2
    #     self.buckets[index]['size'] = self.buckets[index]['high'] - self.buckets[index]['low']
    #     self.buckets.insert(index + 1, bucket2)
    #
    # def mergebuckets(self, index):
    #     self.buckets[index]['frequency'] = self.buckets[index]['frequency'] + self.buckets[index + 1]['frequency']
    #     self.buckets[index]['high'] = self.buckets[index + 1]['high']
    #     self.buckets[index]['size'] = self.buckets[index]['high'] - self.buckets[index]['low']
    #     del self.buckets[index + 1]
    #
    # def candidateMergePair(self):
    #     count = 0
    #     index = None
    #     for i in range(0, self.numbuckets - 1):
    #         if self.buckets[i]['regular'] == False and self.buckets[i + 1]['regular'] == False and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.split:
    #             if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] > count:
    #                 count = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
    #                 index = i
    #         elif self.buckets[i]['regular'] == False and self.buckets[i + 1]['regular'] == True and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.split:
    #             if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] > count:
    #                 count = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
    #                 index = i
    #         elif self.buckets[i]['regular'] == False and self.buckets[i + 1]['regular'] == True and self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] < self.split:
    #             if self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency'] > count:
    #                 count = self.buckets[i]['frequency'] + self.buckets[i + 1]['frequency']
    #                 index = i
    #     return index
    #
    # def candidatesplitbucket(self, gamma):
    #     count = 0
    #     index = None
    #     for i in range(0, self.numbuckets):
    #         if self.buckets[i]['regular'] == True and self.buckets[i]['frequency'] >= 2 * (self.merge + 1):
    #             if self.buckets[i]['frequency'] > count:
    #                 count = self.buckets[i]['frequency']
    #                 index = i
    #         elif self.buckets[i]['regular'] == False and self.buckets[i]['frequency'] <= self.split / (2 + gamma) and self.buckets[i]['frequency'] >= 2(self.merge + 1):
    #             if self.buckets[i]['frequency'] > count:
    #                 count = self.buckets[i]['frequency']
    #                 index = i
    #     return index

    def plot_histogram(self, attr, buckets):
        """Plots the histogram."""
        bins = []
        frequency = []
        for bucket in self.regular:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
        bins.append(bucket['high'])

        frequency = np.array(frequency)
        bins = np.array(bins)

        widths = bins[1:] - bins[:-1]

        plt.bar(bins[:-1], frequency, width=widths, color='#348ABD')

        # if len(self.singular) != 0:
        #     singbins = []
        #     singfrequency = []
        #     for bucket in self.singular:
        #         singbins.append(bucket['low'])
        #         singfrequency.append(bucket['frequency'])
        #     singbins.append(bucket['high'])
        #
        #     singfrequency = np.array(singfrequency)
        #     singbins = np.array(singbins)
        #     singwidths = singbins[1:] - singbins[:-1]
        #
        #     plt.bar(singbins[:-1], singfrequency, width=singwidths)#, color='#348ABD')

        print len(self.regular), len(self.singular)
        print "PLOTTING LENGTH"

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

    def plot_buckets(self):
        bins = []
        frequency = []
        for bucket in self.regular:
            bins.append(bucket['low'])
            frequency.append(bucket['frequency'])
        for bucket in self.singular:
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
