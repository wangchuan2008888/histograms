"""
It constructs an equi-width histogram from the dataset given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from collections import Counter
import user_distribution
import json
import os
from scipy import stats
import scipy.interpolate as interpolate
from shutil import copyfile

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

class Control_Histogram(object):



    """
    This class models an instance of a control histogram, which is equi-width and stretches 
    its bucket boundaries to include values that are beyond the leftmost and rightmost buckets.
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
            f = open(outputpath + "//data//controlksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//controlzipf.jpg")
        plt.close()

    def create_histogram(self, attr, batchsize, userbucketsize):
        """Reads through the file and creates the histogram, adding in data points as they are being read."""
        N = 0
        sample = []
        initial = False
        try:
            os.remove(self.outputpath + "//data//controlksstats" + ".json")
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
                    if len(set(sample)) < self.numbuckets:
                        sample.append(value)
                    if len(set(sample)) == self.numbuckets and initial == False:
                        self.create_initial_histogram(N, sample)
                        #self.plot_histogram(attr, self.buckets)
                        #d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        #d.create_distribution(self.buckets)
                        #new_buckets = d.return_distribution()
                        #self.plot_histogram(attr, new_buckets)
                        initial = True
                    elif initial == True:
                        self.add_datapoint(value)
                        if N % batchsize == 0:
                            print "number read in: " + str(N)
                            #self.plot_histogram(attr, self.buckets)
                            #d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                            #d.create_distribution(self.buckets)
                            #new_buckets = d.return_distribution()
                            #self.plot_histogram(attr, new_buckets)
                            self.compare_histogram(attr, True, N)
                            #f = 0
                            #for i in range(len(self.buckets)):
                            #    f += self.buckets[i]['frequency']
                            #print f, N
                            #assert np.isclose(f, N)
            if len(set(sample)) < self.numbuckets:
                print("ERROR: There are not enough unique values for the number of specified buckets.")
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
            ksstats['cdfstats'] = stats.ks_2samp(realdist, self.inverse_transform_sampling(frequency, binedges, N))
            linear = LinearApproxHist(self.min, self.max, self.buckets, self.numbuckets, cumfreq)
            #here we use the linear approximation of the cdf to create a sample and then compare that to the true dataset
            ksstats['linearcdfstats'] = stats.ks_2samp(realdist, linear.rvs(size=N))
            with open(self.outputpath + "//data//controlksstats" + ".json", 'a+') as ks:
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
        # plt.savefig(self.outputpath + "//img//controlcdf" + str(self.counter) + ".jpg")
        # self.counter += 1
        # plt.close()

    def inverse_transform_sampling(self, frequency, bin_edges, n_samples):
        cum_values = np.zeros(len(bin_edges))
        cum_values[1:] = np.cumsum(frequency) / n_samples
        inv_cdf = interpolate.interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    def create_initial_histogram(self, N, sample):
        """Creates the bucket boundaries based on the first n distinct points present in the sample."""
        sorted_sample = sorted(list(set(sample)), key=float)
        c = Counter(sample)
        r = (max(sorted_sample) + 1) - min(sorted_sample)
        width = r / self.numbuckets
        low = sorted_sample[0]
        for i in range(self.numbuckets):
            self.buckets[i]['size'] = width
            self.buckets[i]['low'] = low
            self.buckets[i]['high'] = low + width
            for j in range(len(sorted_sample)):
                if low <= sorted_sample[j] < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += c[sorted_sample[j]]
            low = self.buckets[i]['high']
        self.buckets[0]['low'] = self.min
        self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        self.buckets[self.numbuckets - 1]['high'] = self.max + 1
        self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        f = 0
        for i in range(len(self.buckets)):
            f += self.buckets[i]['frequency']
        print f, N
        #assert np.isclose(f, N)
    
    def add_datapoint(self, value):
        """Increases the count of the bucket the value belongs in the histogram."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
        elif value >= self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = self.buckets[self.numbuckets - 1]['high'] - self.buckets[self.numbuckets - 1]['low']
        else:
            for i in range(0, self.numbuckets):
                if value >= self.buckets[i]['low'] and value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1

    def plot_histogram(self, attr, buckets):
        """It plots the histogram. """
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
        
        axes.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.title(r'$\mathrm{Control\ Histogram\ of\ ' + attr + '}$')

        with open(self.outputpath + "//data//control" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//control" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
