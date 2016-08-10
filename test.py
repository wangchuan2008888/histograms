import dynamic_histograms.control_histogram
import dynamic_histograms.dc_histogram
import dynamic_histograms.dvo_histogram
import dynamic_histograms.equidepth_histogram
import dynamic_histograms.maxdiff_histogram
import dynamic_histograms.sf_histogram
import dynamic_histograms.spline_histogram
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='Path to dataset')
    parser.add_argument('-attr', required=True, help='The attribute to compute the histogram on')
    parser.add_argument('-histogram', default='control', help='control | dc | dvo | equidepth | maxdiff | sf | spline')
    parser.add_argument('-batchsize', required=True, type=int, help='The size of the batch')
    parser.add_argument('-buckets', required=True, type=int, help='The number of buckets in the histogram')
    parser.add_argument('-userbuckets', required=True, type=int, help='The number of buckets in the histogram that will be displayed to user')
    parser.add_argument('-l', type=float, help='The lambda parameter for equidepth histogram')
    parser.add_argument('-m', type=float, help='Merge threshold for self-tuning & compressed histograms')
    parser.add_argument('-s', type=float, help='Split threshold for self-tuning & compressed histograms')
    parser.add_argument('-alpha', type=float, help='The alpha parameter for some histograms')
    opts = parser.parse_args()

    if opts.histogram == 'control':
        print "### CONTROL HISTOGRAM ###"
        start_time = time.time()
        control = dynamic_histograms.control_histogram.Control_Histogram(opts.dataset, opts.buckets)
        control.create_histogram(opts.attr, opts.batchsize, opts.userbuckets)
        control_time = time.time() - start_time
        print "-------- %f seconds for whole dataset --------" % control_time
    elif opts.histogram == 'dc':
        if opts.m == None:
            print "Please specify the gamma parameter."
            sys.exit(0)
        elif opts.s == None:
            print "Please specify the gamma merge parameter."
            sys.exit(0)
        print "### DYNAMIC COMPRESSED HISTOGRAM ###"
        start_time = time.time()
        dc = dynamic_histograms.dc_histogram.DC_Histogram(opts.dataset, opts.buckets)
        dc.create_histogram(opts.attr, opts.s, opts.m, opts.batchsize)
        dc_time = time.time() - start_time
        print "-------- %f seconds for 3 batches --------" %  dc_time
    elif opts.histogram == 'dvo':
        print "### DYNAMIC V-OPTIMAL HISTOGRAM ###"
        start_time = time.time()
        dvo = dynamic_histograms.dvo_histogram.DVO_Histogram(opts.dataset, opts.buckets)
        dvo.create_dvo_histogram(opts.attr, opts.batchsize)
        dvo_time = time.time() - start_time
        print "-------- %f seconds for 3 batches --------" %  dvo_time
    elif opts.histogram == 'equidepth':
        if opts.l == None:
            print "Please specify the lambda parameter."
            sys.exit(0)
        print "### EQUI-DEPTH HISTOGRAM ###"
        start_time = time.time()
        depth = dynamic_histograms.equidepth_histogram.Equidepth_Histogram(opts.dataset, opts.buckets)
        depth.create_histogram(opts.attr, opts.l, opts.batchsize)
        depth_time = time.time() - start_time
        print "-------- %f seconds for 3 batches --------" %  depth_time
    elif opts.histogram == 'maxdiff':
        print "### MAX-DIFF HISTOGRAM ###"
        start_time = time.time()
        maxdiff = dynamic_histograms.maxdiff_histogram.MaxDiff_Histogram(opts.dataset, opts.buckets)
        maxdiff.create_histogram(opts.attr, opts.batchsize)
        maxdiff_time = time.time() - start_time
        print "-------- %f seconds for 3 batches --------" %  maxdiff_time
    elif opts.histogram == 'sf':
        if opts.alpha == None:
            print "Please specify the alpha parameter."
            sys.exit(0)
        elif opts.m == None:
            print "Please specify the merge threshold parameter."
            sys.exit(0)
        elif opts.s == None:
            print "Please specify the split threshold parameter."
            sys.exit(0)
        print "### SELF-TUNING HISTOGRAM ###"
        start_time = time.time()
        sf = dynamic_histograms.sf_histogram.SF_Histogram(opts.dataset, opts.buckets)
        sf.create_histogram(opts.attr, opts.alpha, opts.m, opts.s, opts.batchsize)
        sf_time = time.time() - start_time
        print "-------- %f seconds for 3 batches --------" %  sf_time
    elif opts.histogram == 'spline':
        print "### SPLINE HISTOGRAM ###"
        start_time = time.time()
        spline = dynamic_histograms.spline_histogram.Spline_Histogram(opts.dataset, opts.buckets)
        spline.create_histogram(opts.attr, opts.batchsize)
        spline_time = time.time() - start_time
        print "-------- %f seconds for 3 batches --------" %  spline_time

if __name__ == '__main__':
    main()
