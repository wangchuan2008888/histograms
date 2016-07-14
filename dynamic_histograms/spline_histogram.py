'''
Given samples, it constructs the appropriate histogram from the sample

Steffani Gomez(smg1)
'''

from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv
from collections import Counter

class Spline_Histogram(object):

    def __init__(self, file, numbuckets):
        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'size': 0,
                'frequency': 0,
                'ff': 0,
                'vv': 0,
                'vf': 0,
                'f': 0,
                'v': 0
            })
        self.buckets = buckets