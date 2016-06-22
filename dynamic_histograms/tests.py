from django.test import TestCase

from data_loader import CSVHelper, SampleTaker

# Create your tests here.

class SampleTakerTests(TestCase):

    # tests that the initial sample contains 50 rows
    def test_initial_sample(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertEqual(len(s.index), 50)

    # tests that subsequent sampling adds a batch of 50 rows to the sample  
    def test_subsequent_sampling(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertEqual(len(s.index), 50)
        s = sampletaker.sample_on_attribute('mpg', s)
        self.assertEqual(len(s.index), 100)

    # tests that if a sample is given on a different attribute than the one passed in,
    # then an error is raised
    def test_different_attr(self):
        csvhelper = CSVHelper()
        f = csvhelper.dataframe_from_csv('dynamic_histograms/data/cars.csv')
        sampletaker = SampleTaker(f)
        s = sampletaker.sample_on_attribute('mpg')
        self.assertRaises(AttributeError, sampletaker.sample_on_attribute, 'origin', s)
