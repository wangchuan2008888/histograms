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
    def sample_on_attribute(self, attr, sample=None, rangelow=None, rangehigh=None):
        # if this is the first sample to be taken from the dataset for the initial histogram
        if sample is None:
        #if rangelow is None and rangehigh is None:
            self.frame = self.frame.sort_values(attr, inplace=False, kind='quicksort', na_position='last')
            minimum = min(self.frame[attr])
            maximum = max(self.frame[attr])
            frame = self.frame[attr][:50]
            sample = frame[np.isfinite(frame)]
            return (pd.DataFrame(sample), minimum, maximum)
        else:
            if (attr != sample.columns.values[0]):
                raise AttributeError('The attribute of the sample does not match the attribute given as an argument.')
            if rangelow == None or rangehigh == None:
                raise ValueError('The function call is either missing the range low or the range high for the sample.')
            #data = list(self.frame[attr])
            sample = []
            for x in self.frame[attr]:
                if x >= rangelow and x < rangehigh:
                    sample.append(x)
                if x >= rangehigh:
                    break
            return pd.DataFrame(sample, index=None, columns=[attr])
            '''
            length = len(sample.index)
            if self.length - (length * batchnum) < 50:
                frame = self.frame[attr][length * batchnum:(self.length - (length * batchnum))]
                frame = pd.DataFrame(frame[np.isfinite(frame)])
                #sample = sample.append(frame)
                #return sample
                return frame # returns batches of the dataset to the histogram to self-tune
            else:
                frame = self.frame[attr][length * batchnum:((length * batchnum) + 50)]
                frame = pd.DataFrame(frame[np.isfinite(frame)])
                #sample = sample.append(frame)
                #return sample
                return frame # returns batches of the dataset to the histogram to self-tune
            '''
