"""
It constructs an equi-width histogram from an underlying histogram distribution that should
be much easier to read and interpret.

Steffani Gomez
"""
from __future__ import division

class User_Distribution(object):

    def __init__(self, minimum, maximum, size):
        buckets = []
        width = (maximum - minimum) / size
        low = minimum
        for i in range(0, size):
            buckets.append({
                'low': low,
                'high': low + width,
                'size': width,
                'frequency': 0
            })
            low += width
        self.buckets = buckets
        self.counter = 0
        self.min = minimum
        self.max = maximum
        self.numbuckets = size

    def sample_original_distribution(self, low, high, buckets, width):
        frequencies = []
        for i in range(0, len(buckets)):
            if buckets[i]['low'] == low and buckets[i]['high'] == high:
                # print "SPECIAL CASE"
                return [(buckets[i]['frequency'], 1)]
            elif buckets[i]['low'] < low and buckets[i]['high'] > low and buckets[i]['high'] < high: # when the bucket overlaps with the specific range
                # print "INTERSECTING CASE"
                frequencies.append((buckets[i]['frequency'], (buckets[i]['high'] - low) / buckets[i]['size']))
            elif buckets[i]['low'] <= low and buckets[i]['high'] >= high:
                # print "POSSIBLY GREATER THAN BUCKET"
                frequencies.append((buckets[i]['frequency'], width / buckets[i]['size']))
            elif buckets[i]['low'] > low and buckets[i]['high'] < high:
                # print "SMALLER BUCKET"
                frequencies.append((buckets[i]['frequency'], 1))
            elif buckets[i]['low'] > low and buckets[i]['low'] < high and buckets[i]['high'] >= high:
                # print "OTHER INTERSECTING CASE"
                frequencies.append((buckets[i]['frequency'], (high - buckets[i]['low']) / buckets[i]['size']))
            #elif buckets[i]['low'] == low and buckets[i]['high'] > high:
        return frequencies


    def sum_freq(self, frequencies):
        freq = 0
        for i in range(0, len(frequencies)):
            freq += frequencies[i][0] * frequencies[i][1]
        return freq

    def create_distribution(self, buckets):
        for i in range(0, self.numbuckets):
            frequencies = self.sample_original_distribution(self.buckets[i]['low'], self.buckets[i]['high'], buckets, self.buckets[i]['size'])
            self.buckets[i]['frequency'] = self.sum_freq(frequencies)

    def return_distribution(self):
        return self.buckets