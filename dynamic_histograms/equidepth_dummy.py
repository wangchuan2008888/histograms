import equidepth_histogram

h = equidepth_histogram.Equidepth_Histogram('data/distributions.csv', 10)
h.create_histogram('chi', l=0, batchsize=1000)