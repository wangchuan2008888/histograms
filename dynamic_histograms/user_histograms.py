"""
It constructs an equi-width histogram from an underlying histogram distribution that should
be much easier to read and interpret.

Steffani Gomez
"""
from __future__ import division
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class User_Histogram(object):

    def __init__(self, minimum, maximum, numbuckets):
        self.min = minimum
        self.max = maximum
        self.numbuckets = numbuckets
        buckets = []
        range = maximum - minimum
        width = range / numbuckets
        low = minimum
        for i in range(0, numbuckets):
            buckets.append({
                'low': low,
                'high': low + width,
                'size': width,
                'frequency': 0
            })
            low += width
        self.min = float("inf")
        self.max = float("-inf")
        self.buckets = buckets
        self.counter = 0

    def sample_distribution(self, low, high, buckets, width):
        frequencies = []
        for i in range(0, self.numbuckets):
            if buckets[i]['low'] == low and buckets[i]['high'] == high:
                return [(buckets[i]['frequency'], 1)]
            elif buckets[i]['low'] < low and buckets[i]['high'] < high: # when the bucket overlaps with the specific range
                frequencies.append((buckets[i]['frequency'], (buckets[i]['high'] - low) / buckets[i]['size']))
            elif buckets[i]['low'] =< low and buckets[i]['high'] >= high:
                frequencies.append((buckets[i]['frequency'], width / buckets[i]['range']))
            elif buckets[i]['low'] > low and buckets[i]['high'] < high:
                frequencies.append((buckets[i]['frequency'], 1))
            elif buckets[i]['low'] > low and buckets[i]['high'] >= high:
                frequencies.append((buckets[i]['frequency'], (high - buckets[i]['low']) / buckets[i]['size']))
            #elif buckets[i]['low'] == low and buckets[i]['high'] > high:

        return frequencies


    def sum_freq(self, frequencies):
        freq = 0
        for i in range(0, len(frequencies)):
            freq += frequencies[0] * frequencies[1]
        return freq

    def create_histogram(self, buckets):
        pass