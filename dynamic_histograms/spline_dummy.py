import spline_histogram

h = spline_histogram.Spline_Histogram('data/cars.csv', 10)
h.create_histogram('mpg', batchsize=1000)