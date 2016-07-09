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

class DVO_Histogram(object):

    def __init__(self, fil, numbuckets):
        self.file = file
        self.numbuckets = numbuckets
        buckets = []
        for i in range(0, self.numbuckets):
            buckets.append({
                'low': 0,
                'high': 0,
                'frequency': 0,
                'size': 0,
            })
        self.buckets = buckets

    