import histograms

h = histograms.DC_Histogram('data/cars.csv', 20)
h.create_dc_histogram('mpg', 0.000001)
h.plot_dc_histogram('mpg')