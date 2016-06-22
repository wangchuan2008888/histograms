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
            frame = self.frame[attr][:50]
            sample = frame[np.isfinite(frame)]
            return pd.DataFrame(sample)
        else:
            if (attr != sample.columns.values[0]):
                raise AttributeError('The attribute of the sample does not match the attribute given as an argument.')
            length = len(sample.index)
            if self.length - length < 50:
                frame = self.frame[attr][length:(self.length - length)]
                frame = pd.DataFrame(frame[np.isfinite(frame)])
                sample = sample.append(frame)
                return sample
            else:
                frame = self.frame[attr][length:(length + 50)]
                frame = pd.DataFrame(frame[np.isfinite(frame)])
                sample = sample.append(frame)
                return sample
