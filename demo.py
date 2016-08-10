import dynamic_histograms.control_histogram
import dynamic_histograms.dc_histogram
import dynamic_histograms.dvo_histogram
import dynamic_histograms.equidepth_histogram
import dynamic_histograms.maxdiff_histogram
import dynamic_histograms.sf_histogram
import dynamic_histograms.spline_histogram
import time

print "### CONTROL HISTOGRAM ###"
start_time = time.time()
control = dynamic_histograms.control_histogram.Control_Histogram('dynamic_histograms/data/distributions.csv', buckets)
control.create_histogram('norm', batchsize=100000, userbuckets=25)
control_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" % control_time

print "### DYNAMIC COMPRESSED HISTOGRAM ###"
start_time = time.time()
dc = dynamic_histograms.dc_histogram.DC_Histogram('dynamic_histograms/data/distributions.csv', buckets)
dc.create_histogram('norm', gamma=0.5, gammam=0.5, batchsize=100000)
dc_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  dc_time

print "### EQUI-DEPTH HISTOGRAM ###"
start_time = time.time()
depth = dynamic_histograms.equidepth_histogram.Equidepth_Histogram('dynamic_histograms/data/distributions.csv', buckets)
depth.create_histogram('norm', l=0, batchsize=100000)
depth_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  depth_time

print "### MAX-DIFF HISTOGRAM ###"
start_time = time.time()
maxdiff = dynamic_histograms.maxdiff_histogram.MaxDiff_Histogram('dynamic_histograms/data/distributions.csv', buckets)
maxdiff.create_histogram('norm', batchsize=100000)
maxdiff_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  maxdiff_time

print "### SELF-TUNING HISTOGRAM ###"
start_time = time.time()
sf = dynamic_histograms.sf_histogram.SF_Histogram('dynamic_histograms/data/distributions.csv', buckets)
sf.create_histogram('norm', alpha=0.5, m=0.01, s=0.1, batchsize=100000)
sf_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  sf_time

print "### SPLINE HISTOGRAM ###"
start_time = time.time()
spline = dynamic_histograms.spline_histogram.Spline_Histogram('dynamic_histograms/data/distributions.csv', buckets)
spline.create_histogram('norm', batchsize=100000)
spline_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  spline_time

print "### DYNAMIC V-OPTIMAL HISTOGRAM"
start_time = time.time()
dvo = dynamic_histograms.dvo_histogram.DVO_Histogram('dynamic_histograms/data/distributions.csv', buckets)
dvo.create_dvo_histogram('norm', batchsize=100000)
dvo_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  dvo_time

print "Control histogram time: %f " % control_time
print "Dynamic compressed histogram time: %f " % dc_time
print "Dynamic v-optimal histogram time: %f " % dvo_time
print "Equi-depth histogram time: %f " % depth_time
print "Max-diff histogram time: %f " % maxdiff_time
print "Self-tuning histogram time: %f " % sf_time
print "Spline histogram time: %f " % spline_time
