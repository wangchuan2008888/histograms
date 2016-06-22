'''
Loads csv files containing datasets and loads them into pandas 
data frames and samples from this

Steffani Gomez(smg1)
'''
import csv
import pandas as pd
import numpy as np

class CSVHelper(object):
    # create a data frame object with 0-index rows a csv file
    def dataframe_from_csv(self, path):
        frame = pd.read_csv(path, header=0, parse_dates=True, encoding='utf-8-sig')
        return frame

class SampleTaker(object):
    # the class must be instantiated with a DataFrame
    def __init__(self, frame):
        self.frame = frame
        self.length = len(frame.index)

    # returns a sample of the dataset on that attribute
    def sample_on_attribute(self, attr, sample=None):
        # if this is the first sample to be taken from the dataset for the initial histogram
        if (sample is None):
            frame2 = self.frame[attr][:50]
            sample = frame2[np.isfinite(frame2)]
        else:
            length = len(sample.index)
        return sample
        