import data_loader
import histograms

csvhelper = data_loader.CSVHelper()
f = csvhelper.dataframe_from_csv('data/cars.csv')
sampletaker = data_loader.SampleTaker(f)
s = sampletaker.sample_on_attribute('mpg')
sample = s[0]
h = histograms.Histogram(f, s[1], s[2])
h.create_initial_sf_histogram('mpg')
h.plot_sf_histogram('mpg')
for i in range(0, 10):
    low = h.buckets[i]['low']
    high = h.buckets[i]['high']
    sample = sampletaker.sample_on_attribute('mpg', sample, rangelow=low, rangehigh=high)
    size = len(sample)
    h.updateFreq(low, high, size, 0.5)
    h.plot_sf_histogram('mpg')
h.restructureHist(0.01, 0.1)

print "### FINAL HISTOGRAM AFTER RESTRUCTURING ###"
print h.buckets
print "### FINITO ###"
h.plot_sf_histogram('mpg')

for i in range(0, h.numbuckets):
    low = h.buckets[i]['low']
    high = h.buckets[i]['high']
    sample = sampletaker.sample_on_attribute('mpg', sample, rangelow=low, rangehigh=high)
    size = len(sample)
    h.updateFreq(low, high, size, 0.5)
    h.plot_sf_histogram('mpg')

h.restructureHist(0.01, 0.1)
print h.buckets
h.plot_sf_histogram('mpg')
