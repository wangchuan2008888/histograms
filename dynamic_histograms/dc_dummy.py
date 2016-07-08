import histograms

h = histograms.DC_Histogram('data/distributions.csv', 250)
h.create_dc_histogram('chi', 0.000001, batchsize=5000)
h.plot_dc_histogram('chi')