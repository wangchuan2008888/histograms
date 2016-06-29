from __future__ import division
from django.test import TestCase
from data_loader import CSVHelper, SampleTaker
from histograms import Histogram

# Create your tests here.

class SampleTakerTests(TestCase):

    # tests that the initial sample returns minimum and maximum as well as a test sample
    def test_initial_sample(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertEqual(len(s[0].index), 50)
        self.assertEqual(s[1], 9.0)

    # tests that subsequent sampling in ranges returns the proper number of values that is in that range
    def test_subsequent_sampling(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertEqual(len(s[0].index), 50)
        s = sampletaker.sample_on_attribute('mpg', s[0], 9.0, 13.0)
        self.assertEqual(len(s.index), 334)

    # tests that if a sample is given on a different attribute than the one passed in,
    # then an error is raised
    def test_different_attr(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertRaises(AttributeError, sampletaker.sample_on_attribute, 'origin', s[0], 9.0, 13.0)

    # tests that a ValueError is raised if one of the ranges is missing
    def test_missing_range(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertRaises(ValueError, sampletaker.sample_on_attribute, 'mpg', s[0], 9.0)

class HistogramTest(TestCase):

    # makes sure that the buckets of the initial histogram create a histogram that is equi-depth and equi-width    
    def test_inital_sf_histogram(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        h.create_initial_sf_histogram('mpg')
        size = h.buckets[0]['size']
        frequency = h.buckets[0]['frequency']
        for i in range(0, h.numbuckets):
            self.assertEqual(size, h.buckets[i]['size'])
            self.assertEqual(frequency, h.buckets[i]['frequency'])

    # tests to make sure the frequency updater method is woking properly
    def test_update_freq(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        h.create_initial_sf_histogram('mpg')
        low = h.buckets[0]['low']
        high = h.buckets[0]['high']
        sample = sampletaker.sample_on_attribute('mpg', sample, rangelow=low, rangehigh=high)
        size = len(sample)
        h.updateFreq(low, high, size, 0.5)
        self.assertEqual(667.0, h.buckets[0]['frequency'])
    
    # this method checks to make sure that buckets are split properly: that the bucket is split evenly across 
    # both range and frequency
    def test_splitting_buckets(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        h.create_initial_sf_histogram('mpg')
        low = h.buckets[0]['low']
        high = h.buckets[0]['high']
        sample = sampletaker.sample_on_attribute('mpg', sample, rangelow=low, rangehigh=high)
        size = len(sample)
        h.updateFreq(low, high, size, 0.5)
        prevlength = h.numbuckets
        prevfreq = h.buckets[0]['frequency']
        prevrange = h.buckets[0]['size']
        prevhigh = h.buckets[0]['high']
        h.splitbucket(h.buckets[0], 2, h.buckets[0]['frequency'])
        self.assertEqual(h.numbuckets, prevlength + 2) # checks that two more buckets were added
        low = h.buckets[0]['low']
        high = low + prevrange / 3
        for i in range(0, 3):
            self.assertEqual(h.buckets[i]['frequency'], prevfreq / 3)
            self.assertEqual(h.buckets[i]['low'], low)
            self.assertEqual(h.buckets[i]['high'], high)
            low = high
            if i == 1:
                high = prevhigh
            else:
                high = low + prevrange / 3
    
    # tests to make sure the equalBuckets method from the Histograms class is working properly
    def test_equalBuckets(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        b1 = {
            'high': 1,
            'low': 2,
            'frequency': 3,
            'size': 1,
            'merge': False
        }
        b2 = {
            'high': 1,
            'low': 2,
            'frequency': 3,
            'size': 1,
            'merge': False
        }
        self.assertEqual(True, h.equalBuckets(b1, b2))
        b2['frequency'] = 1
        self.assertEqual(False, h.equalBuckets(b1, b2))

    def test_checkBucketLists(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        b1 = {
            'high': 2,
            'low': 1,
            'frequency': 3,
            'size': 1,
            'merge': False
        }
        b2 = {
            'high': 2,
            'low': 1,
            'frequency': 3,
            'size': 1,
            'merge': False
        }
        l1 = [b1]
        l2 = [b1]
        self.assertEqual(True, h.checkBucketLists(l1, l2))
        l1.append(b2)
        l2.append(b2)
        self.assertEqual(True, h.checkBucketLists(l1, l2))
        b3 = {
            'high': 8,
            'low': 3,
            'frequency': 2,
            'size': 5,
            'merge': False
        }
        l2.append(b3)
        self.assertEqual(False, h.checkBucketLists(l1, l2))
        l1.append(b1)
        self.assertEqual(False, h.checkBucketLists(l1, l2))

    # this method tests that the mergeruns method in the Histograms class is working properly and joins the 
    # two lists of buckets and replaces the original two bucket runs.
    # done simply, still need to test complicated case where merging runs of multiple buckets not just two
    def test_mergeruns(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        b1 = {
            'high': 2,
            'low': 1,
            'frequency': 3,
            'size': 1,
            'merge': False
        }
        b2 = {
            'high': 4,
            'low': 2,
            'frequency': 5,
            'size': 2,
            'merge': False
        }
        b = [[b1], [b2]]
        bprime = h.mergeruns(b, b[0], b[1])
        self.assertEqual(2, len(bprime[0])) # makes sure the list is merged. meaning there should be 2 buckets
        self.assertEqual(bprime[0][0]['low'], b1['low'])
        self.assertEqual(bprime[0][1]['high'], b2['high'])
        self.assertEqual(bprime[0][0]['frequency'], b1['frequency'])
        self.assertEqual(bprime[0][1]['frequency'], b2['frequency'])

    # tests that the mergebuckets method in the Histograms class merges buckets together properly
    # done simply, still need to test complicated case where merging multiple buckets not just two
    def test_mergebuckets(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        sample = s[0]
        h = Histogram(f, s[1], s[2])
        h.create_initial_sf_histogram('mpg')
        bucketruns = []
        for b in h.buckets:
            bucketruns.append([b])
        totalfreq = h.buckets[0]['frequency'] + h.buckets[1]['frequency']
        bucketruns = h.mergeruns(bucketruns, bucketruns[0], bucketruns[1])
        for b in bucketruns:
            if len(b) != 1:
                h.mergebuckets(b)
        self.assertEqual(9, h.numbuckets)
        self.assertEqual(9.0, h.buckets[0]['low'])
        self.assertEqual(17.0, h.buckets[0]['high'])
        self.assertEqual(totalfreq, h.buckets[0]['frequency'])
        self.assertEqual(8, h.buckets[0]['size'])
        self.assertEqual(False, h.buckets[0]['merge'])
