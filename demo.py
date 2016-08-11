import dynamic_histograms.control_histogram
import dynamic_histograms.dc_histogram
import dynamic_histograms.dvo_histogram
import dynamic_histograms.equidepth_histogram
import dynamic_histograms.maxdiff_histogram
import dynamic_histograms.sf_histogram
import dynamic_histograms.spline_histogram
import time
import os
from shutil import copyfile

buckets = 500
batchsize = 100000
userbucketsize = 40
attr = 'bimodal'

# make sure output directory exists
outputpath = 'output//' + attr + '//' + str(batchsize) + '_' + str(buckets) + '_' + str(userbucketsize)
if not os.path.exists(outputpath + '//img'):
    os.makedirs(outputpath + '//img')
    
# write index.html by going through the output directory    
copyfile('template.html', outputpath + '//template.html')
with open('output//index.html', 'w') as f:
    f.write('<!DOCTYPE html>\n')
    f.write('<html lang=\"en-US\">\n')
    for maind in [d for d in os.listdir('output') if not os.path.isfile(os.path.join('output', d))]:
        print maind
        f.write('<h2>' + maind + '</h2>\n')
        for subd in [d for d in os.listdir('output//' + maind) if not os.path.isfile(os.path.join('output//' + maind, d))]:
            f.write('<a href=\"' + maind + '//' + subd + '//template.html' + '\">' + subd + '</a><br/>\n')
    f.write('</html>\n')


print "### CONTROL HISTOGRAM ###"
start_time = time.time()
control = dynamic_histograms.control_histogram.Control_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
control.create_histogram(attr, batchsize=batchsize, userbucketsize=userbucketsize)
control_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" % control_time

print "### DYNAMIC COMPRESSED HISTOGRAM ###"
start_time = time.time()
dc = dynamic_histograms.dc_histogram.DC_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
dc.create_histogram(attr, gamma=0.5, gammam=0.5, batchsize=batchsize, userbucketsize=userbucketsize)
dc_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  dc_time

print "### EQUI-DEPTH HISTOGRAM ###"
start_time = time.time()
depth = dynamic_histograms.equidepth_histogram.Equidepth_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
depth.create_histogram(attr, l=0, batchsize=batchsize, userbucketsize=userbucketsize)
depth_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  depth_time

print "### MAX-DIFF HISTOGRAM ###"
start_time = time.time()
maxdiff = dynamic_histograms.maxdiff_histogram.MaxDiff_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
maxdiff.create_histogram(attr, batchsize=batchsize, userbucketsize=userbucketsize)
maxdiff_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  maxdiff_time

print "### SELF-TUNING HISTOGRAM ###"
start_time = time.time()
sf = dynamic_histograms.sf_histogram.SF_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
sf.create_histogram(attr, alpha=0.5, m=0.0025, s=0.1, batchsize=batchsize, userbucketsize=userbucketsize)
sf_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  sf_time

print "### SPLINE HISTOGRAM ###"
start_time = time.time()
spline = dynamic_histograms.spline_histogram.Spline_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
spline.create_histogram(attr, batchsize=batchsize, userbucketsize=userbucketsize)
spline_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  spline_time

print "### DYNAMIC V-OPTIMAL HISTOGRAM"
start_time = time.time()
dvo = dynamic_histograms.dvo_histogram.DVO_Histogram('dynamic_histograms/data/distributions.csv', buckets, outputpath)
dvo.create_histogram(attr, batchsize=batchsize, userbucketsize=userbucketsize)
dvo_time = time.time() - start_time
print "-------- %f seconds for 3 batches --------" %  dvo_time

print "Control histogram time: %f " % control_time
print "Dynamic compressed histogram time: %f " % dc_time
print "Dynamic v-optimal histogram time: %f " % dvo_time
print "Equi-depth histogram time: %f " % depth_time
print "Max-diff histogram time: %f " % maxdiff_time
print "Self-tuning histogram time: %f " % sf_time
print "Spline histogram time: %f " % spline_time
