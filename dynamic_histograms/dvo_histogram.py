from __future__ import print_function

"""
It constructs a dynamic v-optimal histogram from the dataset given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
#import pandas as pd
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

class DVO_Histogram(object):

    """
    This class models an instance of a v-optimal histogram, which chooses boundaries that minimize the variance
    between the source parameter values, which are the values in this case.
    """

    def __init__(self, file, numbuckets, outputpath):

        """
        Initiates an instance of the class with a csv file containing the dataset and the number 
        of buckets the histogram should have. 
        """
        self.outputpath = outputpath
        self.file = file
        self.numbuckets = int(numbuckets / 2)
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'size': 0,
                'frequency': 0,
                'leftcounter': 0,
                'rightcounter': 0
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
            print ("zipf parameter" + str(parameter))
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
            f = open(outputpath + "//data//dvoksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//dvozipf.jpg")
        plt.close()

    def plot_histogram(self, attr, buckets, voptimal):
        """Plots the histogram."""
        bins = []
        frequency = []
        if voptimal:
            for bucket in buckets:
                bins.append(bucket['low'])
                frequency.append(bucket['leftcounter'])
                bins.append(bucket['low'] + (bucket['size'] / 2))
                frequency.append(bucket['rightcounter'])
            bins.append(bucket['high'])
        else:
            for bucket in buckets:
                bins.append(bucket['low'])
                frequency.append(bucket['frequency'])

            # MIGHT NEED TO CHANGE THIS BACK FOR ACCURACY PURPOSES, WHEN/IF I DO, USE 250 BUCKETS NOT 500


            #frequency.append(bucket['leftcounter'])
            #bins.append(bucket['low'] + (bucket['size'] / 2))
            #frequency.append(bucket['rightcounter'])
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
        plt.title(r'$\mathrm{Dynamic\ V-Optimal\ Histogram\ of\ ' + attr + '}$')
        
        with open(self.outputpath + "//data//dvo" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//dvo" + str(self.counter) + ".jpg")
        plt.close()
        self.counter += 1

    def create_histogram(self, attr, batchsize, userbucketsize):
        """Reads in data from the file, creating a new bucket if the value is beyond it, choosing the best bucket to merge
        and merging those. Otherise it increments the appropriate bucket count and if the error of splitting a bucket is greater
        than the error of splitting it and merging it, then the best bucket is split and the best buckets are merged."""
        N = 0
        sample = []
        initial = False
        try:
            os.remove(self.outputpath + "//data//dvoksstats" + ".json")
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
                    sorted_sample = sorted(list(set(sample)), key=float)
                    c = Counter(sample)
                    for i in range(0, self.numbuckets):
                        self.buckets[i]['low'] = sorted_sample[i]
                        if i == self.numbuckets - 1:
                            self.buckets[i]['high'] = sorted_sample[i] + 1
                        else:
                            self.buckets[i]['high'] = sorted_sample[i + 1]
                        self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                        for j in range(len(sorted_sample)):
                            if sorted_sample[j] > self.buckets[i]['high']:
                                break
                            elif self.buckets[i]['low'] <= sorted_sample[j] < self.buckets[i]['low'] + (self.buckets[i]['size'] / 2):
                                self.buckets[i]['leftcounter'] += c[sorted_sample[j]]
                            elif self.buckets[i]['low'] + (self.buckets[i]['size'] / 2) <= sorted_sample[j] < self.buckets[i]['high']:
                                self.buckets[i]['rightcounter'] += c[sorted_sample[j]]
                        self.buckets[i]['frequency'] = self.buckets[i]['leftcounter'] + self.buckets[i]['rightcounter']
                    self.buckets[0]['low'] = self.min
                    self.buckets[0]['size'] = self.buckets[0]['high'] - self.buckets[0]['low']
                    self.buckets[self.numbuckets - 1]['high'] = self.max + 1
                    self.buckets[self.numbuckets - 1]['size'] = self.max + 1 - self.buckets[self.numbuckets - 1]['low']
                    f = 0
                    for i in range(len(self.buckets)):
                        f += self.buckets[i]['frequency']
                    print(f, N, "created initial histogram!")
                    assert np.isclose(f, N)
                    self.plot_histogram(attr, self.buckets, True)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets, False)
                    initial = True
                elif initial == True:
                    self.add_datapoint(float(row[attr_index]))
                    if N % batchsize == 0:
                        print ("number read in: " + str(N))
                        self.plot_histogram(attr, self.buckets, True)
                        d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                        d.create_distribution(self.buckets)
                        new_buckets = d.return_distribution()
                        self.plot_histogram(attr, new_buckets, False)
                        self.compare_histogram(attr, False)
                        f = 0
                        for i in range(len(self.buckets)):
                            f += self.buckets[i]['frequency']
                        print(f, N)
                        assert np.isclose(f, N)
                        assert len(self.buckets) == self.numbuckets
            if len(set(sample)) < self.numbuckets:
                print("ERROR: There are not enough unique values for the number of specified buckets.")
        self.compare_histogram(attr, False)

    def compare_histogram(self, attr, end):
        frequency = []
        binedges = []
        for bucket in self.buckets:
            binedges.append(bucket['low'])
            frequency.append(bucket['leftcounter'])
            binedges.append(bucket['low'] + (bucket['size'] / 2))
            frequency.append(bucket['rightcounter'])
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
            ksstats['linearcdstats'] = stats.kstest(realdist, lambda x: self.callable_linearcdf(x, cumfreq), N=len(realdist), alternative='two-sided')
            with open(self.outputpath + "//data//dvoksstats" + ".json", 'a+') as ks:
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
        plt.savefig(self.outputpath + "//img//dvocdf" + str(self.counter) + ".jpg")
        self.counter += 1
        plt.close()

    def callable_cdf(self, x, cumfreq):
        values = []
        for value in x:
            v = self.cdf(value, cumfreq)
            if v == None:
                print (value, v)
                print (self.min, self.max)
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
        if value >= self.buckets[self.numbuckets - 1]['high']:
            bucket = {
                'low': self.buckets[self.numbuckets - 1]['high'],
                'high': value + 1,
                'leftcounter': 0,
                'rightcounter': 0,
                'frequency': 1,
                'size': value + 1 - self.buckets[self.numbuckets - 1]['high']
            }
            if value < bucket['low'] + (bucket['size'] / 2):
                bucket['leftcounter'] += 1
            else:
                bucket['rightcounter'] += 1
            self.buckets.append(bucket) # borrow one bucket
            index = self.findBestToMerge()
            self.mergebuckets(index)
        elif value < self.buckets[0]['low']:
            bucket = {
                'low': value,
                'high': self.buckets[0]['low'],
                'leftcounter': 1,
                'rightcounter': 0,
                'frequency': 1,
                'size': self.buckets[0]['low'] - value
            }
            self.buckets.insert(0, bucket) # borrow one bucket
            index = self.findBestToMerge()
            self.mergebuckets(index)
            f = 0
        else:
            for i in range(0, self.numbuckets):
                if self.buckets[i]['low'] <= value < self.buckets[i]['high']:
                    if value < self.buckets[i]['low'] + (self.buckets[i]['size'] / 2):
                        self.buckets[i]['leftcounter'] += 1
                        self.buckets[i]['frequency'] += 1
                        break
                    else:
                        self.buckets[i]['rightcounter'] += 1
                        self.buckets[i]['frequency'] += 1
                        break
            s = self.findBestToSplit()
            mindex = self.findBestToMerge()
            f = 0
            if self.bucketError(self.buckets[s]) > self.adjacentbucketsError(self.buckets[mindex], self.buckets[mindex + 1]):
                # split s
                # merge m and m.next
                self.splitbucket(s)
                self.mergebuckets(mindex)

    def mergebuckets(self, index):
        """Merging two buckets into one bucket in the list of buckets."""
        self.buckets[index]['high'] = self.buckets[index + 1]['high']
        self.buckets[index]['size'] = self.buckets[index]['high'] - self.buckets[index]['low']
        self.buckets[index]['leftcounter'] = self.buckets[index]['frequency']
        self.buckets[index]['rightcounter'] = self.buckets[index + 1]['frequency']
        self.buckets[index]['frequency'] = self.buckets[index]['leftcounter'] + self.buckets[index]['rightcounter']
        del self.buckets[index + 1]

    def splitbucket(self, index):
        """Splits a bucket in the list of buckets of the histogram."""
        bucket2 = {
            'low': (self.buckets[index]['size'] / 2) + self.buckets[index]['low'],
            'high': self.buckets[index]['high'],
            'size': self.buckets[index]['size'] / 2,
            'leftcounter': self.buckets[index]['rightcounter'] / 2,
            'rightcounter': self.buckets[index]['rightcounter'] / 2,
            'frequency': self.buckets[index]['rightcounter']
        }
        self.buckets[index]['high'] = (self.buckets[index]['size'] / 2) + self.buckets[index]['low']
        self.buckets[index]['size'] = self.buckets[index]['size'] / 2
        self.buckets[index]['rightcounter'] = self.buckets[index]['leftcounter'] / 2
        self.buckets[index]['leftcounter'] = self.buckets[index]['leftcounter'] / 2
        self.buckets[index]['frequency'] = self.buckets[index]['leftcounter'] * 2
        self.buckets.insert(index + 1, bucket2)

    def bucketError(self, bucket):
        """Calculates the error of a single bucket and returns it."""
        average = np.average([bucket['leftcounter'] + bucket['rightcounter']])
        lefterror = abs(bucket['leftcounter'] - average)
        righterror = abs(bucket['rightcounter'] - average)
        return lefterror + righterror

    def adjacentbucketsError(self, bucket1, bucket2):
        """Calculates the error of two adjacent buckets and returns it."""
        return self.bucketError(bucket1) + self.bucketError(bucket2)

    def findBestToMerge(self):
        """Returns the index of a bucket such that combined with its success is smallest among all the pairs."""
        minimum = float('inf')
        index = None
        for i in range(0, self.numbuckets - 1):
            error = self.adjacentbucketsError(self.buckets[i], self.buckets[i + 1])
            if error < minimum:
                minimum = error
                index = i
        return index
    
    def findBestToSplit(self):
        """Returns the index of the bucket with the highest error."""
        maximum = float('-inf')
        index = None
        for i in range(0, self.numbuckets):
            error = self.bucketError(self.buckets[i])
            if error > maximum:
                maximum = error
                index = i
        return index

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""        
        for i in range(0, len(self.buckets)):
            print ("### bucket " + str(i) + " ###")
            for k, v in self.buckets[i].iteritems():
                print (k, v)
            print ("### END ###")
