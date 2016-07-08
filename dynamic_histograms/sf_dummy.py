import data_loader
import histograms

csvhelper = data_loader.CSVHelper()
f = csvhelper.dataframe_from_csv('data/distributions.csv')
sampletaker = data_loader.SampleTaker(f)
s = sampletaker.sample_on_attribute('norm')
sample = s[0]
h = histograms.SF_Histogram(f, s[1], s[2])
h.create_initial_sf_histogram('norm')
h.plot_sf_histogram('norm')
for i in range(0, 10):
    low = h.buckets[i]['low']
    high = h.buckets[i]['high']
    sample = sampletaker.sample_on_attribute('norm', sample, rangelow=low, rangehigh=high)
    size = len(sample)
    h.updateFreq(low, high, size, 0.5)
    h.plot_sf_histogram('norm')
h.restructureHist(0.01, 0.1)
h.plot_sf_histogram('norm')

for j in range(0, 5):
    for i in range(0, h.numbuckets):
        low = h.buckets[i]['low']
        high = h.buckets[i]['high']
        sample = sampletaker.sample_on_attribute('norm', sample, rangelow=low, rangehigh=high)
        size = len(sample)
        h.updateFreq(low, high, size, 0.5)
        h.plot_sf_histogram('norm')

    h.restructureHist(0.01, 0.1)
    print h.buckets
    h.plot_sf_histogram('norm')
