import control_histogram

h = control_histogram.Control_Histogram('data/distributions.csv', 10)
h.create_histogram('norm', batchsize=1000)