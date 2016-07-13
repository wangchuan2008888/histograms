import equidepth_histogram

h = equidepth_histogram.Equidepth_Histogram('data/cars.csv', 10)
h.create_histogram('mpg', l=0, batchsize=1000)