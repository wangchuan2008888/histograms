import maxdiff_histogram

h = maxdiff_histogram.MaxDiff_Histogram('data/cars.csv', 10)
h.create_histogram('mpg', batchsize=1000)