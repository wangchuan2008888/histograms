"""
It constructs a simple dynamic max-diff histogram rom the dataset given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from collections import Counter
from collections import defaultdict
import operator
import itertools
import random
import user_distribution
import json
import os
from scipy import stats
import scipy.interpolate as interpolate
from shutil import copyfile

upper_factor = 3

class LinearApproxHist(stats.rv_continuous):
    def __init__(self, minimum, maximum, buckets, numbuckets, cumfreq):
        super(LinearApproxHist, self).__init__()
        self.min = minimum
        self.max = maximum
        self.numbuckets = numbuckets
        self.buckets = buckets
        self.cumfreq = cumfreq

    def _cdf(self, x):
        if x <= self.min:
            return 0
        elif x >= self.max:
            return 1
        for i in range(0, self.numbuckets):
            if x >= self.buckets[i]['low'] and x < self.buckets[i]['high']:
                percentage = (x - self.buckets[i]['low']) / self.buckets[i]['size']
                if i > 0:
                    approx = percentage + self.cumfreq[i - 1]
                else:
                    approx = percentage * self.cumfreq[i]
                return approx / self.cumfreq[len(self.cumfreq) - 1]

class MaxDiff_Histogram(object):

    """
    This class models an instance of a max-diff histogram histogram, which is a histogram that sets boundaries on the
    numbuckets - 1 largest differences in area between the values.
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
                'frequency': 0
            })
        self.buckets = buckets
        self.counter = 0
        self.min = float('inf')
        self.max= float('-inf')
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
            f = open(outputpath + "//data//maxdiffksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//maxdiffzipf.jpg")
        plt.close()


    def create_histogram(self, attr, batchsize, userbucketsize):
        """Reads in records from the file, computing the initial histogram and after each batch by finding numbuckets - 1 
        largest differences in area between each value in the sample and setting the boundaries in between these values."""
        N = 0
        sample = []
        initial = False
        skip = 0
        skipcounter = 0
        try:
            os.remove(self.outputpath + "//data//maxdiffksstats" + ".json")
        except OSError:
            pass
        with open(self.file) as f:
            reader = csv.reader(f)
            header = reader.next()
            for i in range(0, len(header)):
                header[i] = unicode(header[i], 'utf-8-sig')
            attr_index = header.index(attr)
            for row in reader:
                try:
                    value = float(row[attr_index])
                except ValueError:
                    value = None
                if value != None:
                    N += 1
                    if value < self.min:
                        self.min = value
                    if value > self.max:
                        self.max = value
                    if len(set(sample)) < self.numbuckets * 2:
                        sample.append(value)
                    if len(set(sample)) == self.numbuckets * 2 and initial == False:
                        self.compute_histogram(sample, N)
                        #self.plot_histogram(attr, self.buckets)
                        #d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        #d.create_distribution(self.buckets)
                        #new_buckets = d.return_distribution()
                        #self.plot_histogram(attr, new_buckets)
                        skip = self.calculateSkip(len(sample))
                        initial = True
                        #freq = 0
                        #for i in range(len(self.buckets)):
                        #    freq += self.buckets[i]['frequency']
                        #print freq, N
                        #assert np.isclose(freq, N)
                    elif initial == True:
                        skipcounter += 1
                        self.add_datapoint(value)
                        if skipcounter == skip:
                            sample = self.maintainBackingSample(value, sample)
                            skip = self.calculateSkip(len(sample))
                            skipcounter = 0
                        if N % batchsize == 0:
                            print "number read in: " + str(N)
                            #self.plot_histogram(attr, self.buckets)
                            #d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                            #d.create_distribution(self.buckets)
                            #new_buckets = d.return_distribution()
                            #self.plot_histogram(attr, new_buckets)
                            self.compute_histogram(sample, N)
                            self.compare_histogram(attr, True, N)
                            #freq = 0
                            #for i in range(len(self.buckets)):
                            #    freq += self.buckets[i]['frequency']
                            #print freq, N
                            #assert np.isclose(freq, N)
            if len(set(sample)) < self.numbuckets * 2:
                print("ERROR: There are not enough unique values for the number of specified buckets.")
            else:
                #self.plot_histogram(attr, self.buckets)
                self.compare_histogram(attr, True, N)

    def compare_histogram(self, attr, end, N):
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
                try:
                    value = float(row[attr_index])
                except ValueError:
                    value = None
                if value != None:
                    realdist.append(value)
        if end:
            ksstats = {}
            # here we use inverse transform sampling to form a distribution from the histogram
            ksstats['cdfstats'] = stats.ks_2samp(realdist, self.inverse_transform_sampling(frequency, binedges, cumfreq[len(cumfreq) - 1]))
            linear = LinearApproxHist(self.min, self.max, self.buckets, self.numbuckets, cumfreq)
            # here we use the linear approximation of the cdf to create a sample and then compare that to the true dataset
            ksstats['linearcdfstats'] = stats.ks_2samp(realdist, linear.rvs(size=cumfreq[len(cumfreq) - 1]))
            with open(self.outputpath + "//data//maxdiffksstats" + ".json", 'a+') as ks:
                json.dump(ksstats, ks)
                ks.write('\n')
        # sorted_data = np.sort(realdist)
        # yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        # plt.grid(True)
        # plt.plot(sorted_data, yvals)
        # step = [0]
        # step.extend(cumfreq / cumfreq[len(cumfreq) - 1])
        # plt.step(binedges[0:], step)
        # plt.plot(binedges[0:], step)
        # plt.legend(['CDF of real data', 'CDF of histogram', 'CDF of linear approx'], loc='lower right')
        # plt.savefig(self.outputpath + "//img//maxdiffcdf" + str(self.counter) + ".jpg")
        # self.counter += 1
        # plt.close()
        
    def inverse_transform_sampling(self, frequency, bin_edges, n_samples):
        cum_values = np.zeros(len(bin_edges))
        cum_values[1:] = np.cumsum(frequency) / n_samples
        inv_cdf = interpolate.interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    def compute_histogram(self, sample, N):
        """Computes the histogram boundaries by finding the numbuckets - 1 largest differences in areas 
        between values in the sample and then arranges the buckets to have the proper frequency."""
        sorted_sample = sorted(list(set(sample)), key=float)#sorted(sample, key=float)
        c = Counter(sorted_sample)
        a = [-1] * (self.numbuckets - 1)
        bucketarea = defaultdict(list)
        for i in range(0, len(sorted_sample) - 2):
            freq = c[sorted_sample[i]]
            spread = sorted_sample[i + 1] - sorted_sample[i]
            area = spread * freq
            area1 = (sorted_sample[i + 2] - sorted_sample[i + 1]) * c[sorted_sample[i + 1]]
            delta = abs(area1 - area)
            if self.checkDifference(delta, a):
                stats = self.addArea(delta, a, bucketarea, i)
                a = stats[0]
                bucketarea = stats[1]
        self.arrangeBuckets(c, a, bucketarea, sorted_sample, N)

    def addArea(self, area, a, bucketarea, index):
        """Adds the area to the list of areas (a) and the dictionary of areas (bucketarea) and the index 
        of the value."""
        length = len(a)
        for i in range(0, len(a)):
            if area >= a[i]:
                bucketarea[area].append(index)
                a[i:i] = [area]
                area = -2
        val = sorted(a, reverse=True)[length:][0]
        final = sorted(a, reverse=True)[0:length]
        if bucketarea.has_key(val):
            if val not in final:
                del bucketarea[val]
            else:
                l = len(bucketarea[val])
                bucketarea[val] = bucketarea[val][:l - 1]
        return final, bucketarea

    def checkDifference(self, area, a):
        """Checks whether the area is greater than any of the areas in the list (a)."""
        for i in range(0, len(a)):
            if area >= a[i]:
                return True
        return False

    def arrangeBuckets(self, counter, areas, bucketarea, sample, N):
        """Arranges the bucket in order by setting the boundaries in order and calculates the 
        frequency for each bucket."""
        boundaries = sorted(bucketarea.items(), key=operator.itemgetter(1))
        low = self.min
        values = bucketarea.values()
        values = list(itertools.chain(*values))
        values = sorted(values)
        for i in range(0, len(values)):
            self.buckets[i]['low'] = low
            highindex = values[i]
            self.buckets[i]['high'] = sample[highindex]
            self.buckets[i]['size'] = sample[highindex] - low
            if sample[highindex] == self.buckets[i]['low']:
                self.buckets[i]['high'] = sample[highindex + 1]
                self.buckets[i]['size'] = sample[highindex + 1] - low
            if low == self.min:
                self.buckets[i]['frequency'] = counter[sample[0]] * N / len(sample) * 2
            else:
                self.buckets[i]['frequency'] = counter[low] * N / len(sample) * 2
            low = self.buckets[i]['high']
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.buckets[self.numbuckets - 1]['low'] = self.buckets[self.numbuckets - 2]['high']
        self.buckets[self.numbuckets - 1]['frequency'] = counter[self.buckets[self.numbuckets - 1]['low']] * N / len(sample) * 2
        self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        f = 0
        for i in range(len(self.buckets)):
            f += self.buckets[i]['frequency']
        #assert np.isclose(f, N)

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

    def add_datapoint(self, value):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - value
        elif value > self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = value + 1 - self.buckets[self.numbuckets - 1]['low']
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1

    def plot_histogram(self, attr, buckets):
        """Plots the histogram."""
        bins = []
        frequency = []
        for i in range(0, len(buckets)):
            bins.append(buckets[i]['low'])
            frequency.append(buckets[i]['frequency'])
        bins.append(buckets[i]['high'])

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
        plt.title(r'$\mathrm{Max-Diff\ Histogram\ of\ ' + attr + '}$')
        
        with open(self.outputpath + "//data//maxdiff" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//maxdiff" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"