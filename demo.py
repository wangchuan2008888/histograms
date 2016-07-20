import dynamic_histograms.control_histogram
import dynamic_histograms.dc_histogram
import dynamic_histograms.dvo_histogram
import dynamic_histograms.equidepth_histogram
import dynamic_histograms.maxdiff_histogram
import dynamic_histograms.sf_histogram
import dynamic_histograms.spline_histogram

print "### CONTROL HISTOGRAM ###"
control = dynamic_histograms.control_histogram.Control_Histogram('dynamic_histograms/data/distributions.csv', 500)
control.create_histogram('norm', batchsize=5000)

print "### DYNAMIC COMPRESSED HISTOGRAM ###"
dc = dynamic_histograms.dc_histogram.DC_Histogram('dynamic_histograms/data/distributions.csv', 500)
dc.create_dc_histogram('norm', alpha=0.000001, batchsize=5000)

print "### DYNAMIC V-OPTIMAL HISTOGRAM ###"
dvo = dynamic_histograms.dvo_histogram.DVO_Histogram('dynamic_histograms/data/distributions.csv', 500)
dvo.create_dvo_histogram('norm', batchsize=5000)

print "### EQUI-DEPTH HISTOGRAM ###"
depth = dynamic_histograms.equidepth_histogram.Equidepth_Histogram('dynamic_histograms/data/distributions.csv', 500)
depth.create_histogram('norm', l=0, batchsize=5000)

print "### MAX-DIFF HISTOGRAM ###"
maxdiff = dynamic_histograms.maxdiff_histogram.MaxDiff_Histogram('dynamic_histograms/data/distributions.csv', 500)
maxdiff.create_histogram('norm', batchsize=5000)

print "### SELF-TUNING HISTOGRAM ###"
sf = dynamic_histograms.sf_histogram.SF_Histogram('dynamic_histograms/data/distributions.csv', 500)
sf.create_histogram('norm', alpha=0.5, m=0.01, s=0.1, batchsize=5000)

print "### SPLINE HISTOGRAM ###"
spline = dynamic_histograms.spline_histogram.Spline_Histogram('dynamic_histograms/data/cars.csv', 50)
spline.create_histogram('mpg', batchsize=2000)
