import dvo_histogram

h = dvo_histogram.DVO_Histogram('data/distributions.csv', 10)
h.create_dvo_histogram('norm', batchsize=5000)