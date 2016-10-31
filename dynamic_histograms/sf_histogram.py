"""
It constructs a dynamic self-tuning histogram from the dataset given.

Steffani Gomez
"""

from __future__ import division
import numpy as np
#import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter
import user_distribution
import json
import os
from scipy import stats
from shutil import copyfile
from collections import Counter

class SF_Histogram(object):

    """
    This class models an instance of a self-tuning histogram, which is a histogram that updates its 
    frequencies with every insertion and restructures the histogram according to frequency variation 
    between the histogram buckets.
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
                'merge': False
            })
        self.min = float("inf")
        self.max = float("-inf")
        self.buckets = buckets
        self.counter = 0

    def zipfdistributiongraph(self, z, alpha, m, s, batchsize, userbucketsize):
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
            self.create_histogram(attr, alpha, m, s, batchsize, userbucketsize)
            f = open(outputpath + "//data//sfksstats.json")
            d = json.loads(f.readline())
            ksstatistics.append(d['cdfstats'][0])
        plt.grid(True)
        plt.plot(zipfparameter, ksstatistics)
        plt.savefig(path + "//img//sfzipf.jpg")
        plt.close()

    def create_initial_histogram(self, sample):
        """Creates the initial histogram from the sample on the atttribute, using only the sample's min and max
        since the intial self-tuning histogram does not look at the data and assumes a frequency of maximum 
        observations / # of buckets for each bucket
        """
        c = Counter(sample)
        sortedsample = sorted(list(set(sample)), key=float)
        low = sortedsample[0]
        high = sortedsample[1]
        for i in range(self.numbuckets):
            self.buckets[i]['low'] = low
            self.buckets[i]['high'] = high
            self.buckets[i]['frequency'] = c[low]
            low = high
            if i >= self.numbuckets - 2:
                high = sortedsample[len(sortedsample) - 1] + 1
            else:
                high = sortedsample[i + 2]
            self.buckets[i]['size'] = abs(self.buckets[i]['high'] - self.buckets[i]['low'])
        # range = math.ceil(self.max - self.min) # want to make sure we capture the maximum element in the last bucket
        # size = math.ceil(range / self.numbuckets)
        # low = self.min
        # high = self.min + size
        # for bucket in self.buckets:
        #     bucket['low'] = low
        #     bucket['high'] = high
        #     bucket['frequency'] = round(len(sample) / self.numbuckets)
        #     bucket['size'] = size
        #     low = high
        #     high += size

    def create_histogram(self, attr, alpha, m, s, batchsize, userbucketsize):
        N = 0
        sample = []
        initial = False
        try:
            os.remove(self.outputpath + "//data//sfksstats" + ".json")
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
                    self.create_initial_histogram(sample)
                    self.plot_histogram(attr, self.buckets)
                    d = user_distribution.User_Distribution(self.min, self.max, userbucketsize)
                    d.create_distribution(self.buckets)
                    new_buckets = d.return_distribution()
                    self.plot_histogram(attr, new_buckets)
                    initial = True
                    f = 0
                    for i in range(len(self.buckets)):
                        f += self.buckets[i]['frequency']
                    assert np.isclose(f, N)
                elif initial == True:
                    self.add_datapoint(float(row[attr_index]), N)
                    if N % batchsize == 0:
                        # we are choosing not to use updateFreq from the pseudocode as that operates over ranges of data
                        # that are accessed by SQL queries. instead we are just using restructureHist after every batch.
                        print "number read in: " + str(N)
                        f = 0
                        for i in range(len(self.buckets)):
                            f += self.buckets[i]['frequency']
                        print f, N
                        assert np.isclose(f, N)
                        self.restructureHist(m, s, N)
                        f = 0
                        for i in range(len(self.buckets)):
                            f += self.buckets[i]['frequency']
                        assert np.isclose(f, N)
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
            if len(set(sample)) < self.numbuckets:
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
            with open(self.outputpath + "//data//sfksstats" + ".json", 'w') as ks:
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
        plt.savefig(self.outputpath + "//img//sfcdf" + str(self.counter) + ".jpg")
        self.counter += 1
        plt.close()

    def callable_cdf(self, x, cumfreq):
        values = []
        for value in x:
            values.append(self.cdf(value, cumfreq))
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

    def sample_on_range(self, sample, rangelow, rangehigh):
        """Returns the sample over the range rangelow-rangehigh."""
        sample = sorted(sample, key=float)
        s = []
        for i in range(0, len(sample)):
            if sample[i] >= rangelow and sample[i] < rangehigh:
                sample.append(sample[i])
            if sample[i] >= rangehigh:
                break
        return s

    def add_datapoint(self, value, N):
        """Adds data points to the histogram, adjusting the end bucket partitions if necessary."""
        if value < self.buckets[0]['low']:
            self.buckets[0]['low'] = value
            self.buckets[0]['frequency'] += 1
            self.buckets[0]['size'] = self.buckets[0]['high'] - value
        elif value >= self.buckets[self.numbuckets - 1]['high']:
            self.buckets[self.numbuckets - 1]['high'] = value + 1
            self.buckets[self.numbuckets - 1]['frequency'] += 1
            self.buckets[self.numbuckets - 1]['size'] = value + 1 - self.buckets[self.numbuckets - 1]['low']
        else:
            for i in range(0, self.numbuckets):
                if self.buckets[i]['low'] <= value < self.buckets[i]['high']:
                    self.buckets[i]['frequency'] += 1
                    break


    # plots a histogram via matplot.pyplot. this is the intial histogram of the self-tuning histogram which is both equi-depth
    # and equi-width (because the intial histogram does not look at the data frequencies)
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
        plt.title(r'$\mathrm{Self-Tuning\ Histogram\ of\ ' + attr + '}$')
        
        with open(self.outputpath + "//data//sf" + str(self.counter) + ".json", 'w') as outfile:
            json.dump(buckets, outfile)
        plt.savefig(self.outputpath + "//img//sf" + str(self.counter) + ".jpg")
        self.counter += 1
        plt.close()


    # alpha is a dampening factor in the range 0.5 to 1 to make sure that bucket frequencies are not
    # modified too much in response to errors, as this may lead to oversensitive
    # or unstable histograms.

    def updateFreq(self, low, high, act, alpha):
        """Updates the frequency for all the buckets in between the range low-high, using alpha as a 
        dampening factor in the range 0.5 to 1 to make sure that bucket frequences are not modified 
        too much in response to errors."""
        b = []
        for i in range(0, self.numbuckets):
            if (self.buckets[i]['low'] < low and self.buckets[i]['high'] > low):
                b.append(i)
            elif (self.buckets[i]['low'] >= low and self.buckets[i]['high'] <= high):
                b.append(i)
            elif (self.buckets[i]['low'] < high and self.buckets[i]['high'] > high):
                b.append(i)
        est = 0
        for i in b:
            est += self.buckets[i]['frequency']
        esterr = act - est
        for i in b:
            frac = (min(high, self.buckets[i]['high']) - max(low, self.buckets[i]['low']) + 1) / (self.buckets[i]['size'] + 1)
            self.buckets[i]['frequency'] = max(self.buckets[i]['frequency'] + (alpha * esterr * frac * (self.buckets[i]['frequency'] / est)), 0)
    
    def maxdiffbucketruns(self, b1, b2):
        minimum = float('-inf')
        b1index = None
        b2index = None
        for i in range(len(b1)):
            for j in range(len(b2)):
                if abs(b1[i]['frequency'] - b2[j]['frequency']) > minimum:
                    minimum = abs(b1[i]['frequency'] - b2[j]['frequency'])
                    b1index = i
                    b2index = j
        return minimum


    def restructureHist(self, m, s, size):
        """the algorithm for restructing histograms. m is a parameter that we call the merge threshold. 
        In most of the experiments, m <= 1% was a suitable choice. 
        s is a parameter that we call the split threshold. In the experiments, we used s=10%."""
        freebuckets = 0
        bucketruns = []
        for b in self.buckets:
            bucketruns.append([b])
        while True:
            maxfreq = []
            for i in range(0, len(bucketruns) - 1):
                tuple = (self.maxdiffbucketruns(bucketruns[i], bucketruns[i + 1]), i, i + 1)
                #localmax = float('-inf')
                #tuple = []
                #for b1 in bucketruns[i]:
                #    for b2 in bucketruns[i + 1]:
                #        diff = abs(b2['frequency'] - b1['frequency'])
                #        if diff > localmax:
                #            localmax = diff
                #            tuple = [localmax, bucketruns[i], bucketruns[i + 1]]
                maxfreq.append(tuple)
            if len(maxfreq) > 0:
                mintuple = min(maxfreq, key=itemgetter(0))
            else:
                break
            if mintuple[0] <= m * size:
                bucketruns = self.mergeruns(bucketruns, mintuple[1], mintuple[2])
                freebuckets += 1
            else:
                break

        if freebuckets == 0:
            return
        
        k = int(round(s * self.numbuckets))

        unmergedbuckets = []
        for i in range(len(self.buckets)):
            if self.buckets[i]['merge'] == False:
                unmergedbuckets.append(self.buckets[i])
        frequencies = [b['frequency'] for b in unmergedbuckets]
        if len(frequencies) > 0 and k > 0:
            frequencies = sorted(frequencies, reverse=True)
            highfrequencies = frequencies[:k]
            #f = pd.Series(frequencies)
            #highfrequencies = list(f.nlargest(k))
            totalfreq = sum(highfrequencies)
            highbuckets = []
            for b in unmergedbuckets:
                if b['frequency'] in highfrequencies and b['merge'] == False:
                    highbuckets.append(b)
                if len(highbuckets) == len(highfrequencies):
                    break

            # merging each run that has more than one bucket in it, meaning those buckets should be merged together
            #print len(bucketruns)
            for l in bucketruns:
                if len(l) != 1:
                    self.mergebuckets(l)

            #print len(self.buckets), self.numbuckets, freebuckets, len(unmergedbuckets)

            # creating dictionary that keeps track of the number of free buckets each bucket that needs to be split gets
            allocation = self.allocatefreebuckets(freebuckets, highbuckets, totalfreq)

            for b in highbuckets:
                self.splitbucket(b, allocation)

        #self.numbuckets = len(self.buckets)
        #print len(self.buckets), self.numbuckets
        assert len(self.buckets) == self.numbuckets

    def allocatefreebuckets(self, numfree, highbuckets, totalfreq):
        # returns a dictionary that defines the number of buckets each bucket in highbuckets should be split into
        percentages = []
        percindexes = []
        allocation = {}
        initial = 0
        fraccount = 0.0
        for i in range(len(highbuckets)):
            frac = highbuckets[i]['frequency'] / totalfreq#totalfreq / highbuckets[i]['frequency']#highbuckets[i]['frequency'] / totalfreq
            fraccount += frac
            percentages.append(frac)
            percindexes.append(i)
            allocation[highbuckets[i]['low']] = int(frac)
            initial += int(frac)
        if initial <= self.numbuckets * 0.25:
            initial = 0
            for i in range(len(highbuckets)):
                frac = highbuckets[i]['frequency'] / totalfreq
                allocation[highbuckets[i]['low']] = int(frac * numfree)
                initial += int(frac * numfree)
        if initial < numfree:
            decpercs = []
            for i in range(len(percentages)):
                decimal = percentages[i] % 1
                decpercs.append(decimal)
            extra = numfree - initial
            highdecs = sorted(decpercs, reverse=True)[:extra]
            for j in range(len(highdecs)):
                index = decpercs.index(highdecs[j])
                allocation[highbuckets[percindexes[index]]['low']] += 1
        elif initial > numfree:
            print "WE FUCKED"
        sum = 0
        for k in allocation.keys():
            sum += allocation[k]
        #print sum, numfree
        assert sum == numfree
        return allocation



    def splitbucket(self, b, allocation):
        """Splits the bucket into the appropriate number and inserts that into the buckets list kept with the histogram.
        numfree - # of free buckets
        totalfreq - total frequency of the buckets that need to be split."""
        f = 0
        reglen = 0
        for i in range(len(self.buckets)):
            f += self.buckets[i]['frequency']
            reglen += 1
        N = f
        #print "before split", N
        numsplit = allocation[b['low']] + 1 # the number of extra buckets
        size = b['size'] / numsplit
        newbuckets = []
        totalfreq = 0.0
        for i in range(len(self.buckets)):
            if self.buckets[i]['low'] != b['low'] and self.buckets[i]['high'] != b['high']:
                newbuckets.append(self.buckets[i])
                totalfreq += self.buckets[i]['frequency']
            elif self.buckets[i]['low'] == b['low'] and self.buckets[i]['high'] == b['high'] and self.buckets[i]['frequency'] == b['frequency']:
                #print self.buckets[i]['frequency'], b['frequency']
                low = self.buckets[i]['low']
                high = low + size
                freq = 0.0
                for j in range(numsplit):
                    newb = {
                        'low': low,
                        'high': high,
                        'frequency': self.buckets[i]['frequency'] / numsplit,
                        'size': high - low,
                        'merge': False
                    }
                    freq += newb['frequency']
                    low = high
                    high = low + size
                    newbuckets.append(newb.copy())
                #print b['low'], b['high'], low, high, freq, b['frequency'], self.buckets[i]['frequency']
                totalfreq += freq
        self.buckets = newbuckets
        #print reglen, len(self.buckets), numsplit, totalfreq
        f = 0
        for i in range(len(self.buckets)):
            f += self.buckets[i]['frequency']
        #print f, N
        assert np.isclose(f, N)

    def mergeruns(self, buckets, b1, b2):
        """Sets the buckets in b1 and b2 to be merged and merges the lists into one list in buckets."""
        for i in range(len(buckets[b1])):
            buckets[b1][i]['merge'] = True
        for i in range(len(buckets[b2])):
            buckets[b2][i]['merge'] = True
        merged = buckets[b1] + buckets[b2]
        newbuckets = []
        prev = len(buckets)
        for i in range(prev):
            if i == b1:
                newbuckets.append(merged)
            elif i == b2:
                pass
            else:
                newbuckets.append(buckets[i])
        new = len(newbuckets)
        assert new < prev
        return newbuckets

    def mergebuckets(self, bucketrun):
        """Merges all the buckets in bucketrun into one bucket and inserting that bucket where all the previous
        unmerged buckets were."""
        oldlen = len(self.buckets)
        mergelen = len(bucketrun)
        buckets = []
        totalfreq = 0
        low = bucketrun[0]['low']
        for b in bucketrun:
            totalfreq += b['frequency']
        high = b['high']
        for i in range(oldlen):
            if self.buckets[i]['low'] == low and self.buckets[i]['high'] == bucketrun[0]['high']:
                self.buckets[i]['high'] = high
                self.buckets[i]['frequency'] = totalfreq
                self.buckets[i]['size'] = self.buckets[i]['high'] - self.buckets[i]['low']
                for j in range(mergelen - 1):
                    del self.buckets[i + 1]
                if i + 1 < len(self.buckets):
                    assert np.isclose(self.buckets[i + 1]['low'], self.buckets[i]['high'])
                break
        assert len(self.buckets) == oldlen - (mergelen - 1)

    def print_buckets(self):
        """Prints the buckets of the histogram, including bucket boundaries and the count of the bucket."""        
        for i in range(0, self.numbuckets):
            print "### bucket " + str(i) + " ###"
            for k, v in self.buckets[i].iteritems():
                print k, v
            print "### END ###"
            