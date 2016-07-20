import dynamic_histograms.control_histogram
import dynamic_histograms.dc_histogram
import dynamic_histograms.dvo_histogram
import dynamic_histograms.equidepth_histogram
import dynamic_histograms.maxdiff_histogram
import dynamic_histograms.sf_histogram
import dynamic_histograms.spline_histogram

control = dynamic_histograms.control_histogram.Control_Histogram('dynamic_histograms/data/distributions.csv', 500)
control.create_histogram('chi', batchsize=5000)

dc = dynamic_histograms.dc_histogram.DC_Histogram('dynamic_histograms/data/distributions.csv', 500)
dc.create_dc_histogram('norm', alpha=0.000001, batchsize=5000)

dvo = dynamic_histograms.dvo_histogram.DVO_Histogram('dynamic_histograms/data/distributions.csv', 500)
dvo.create_dvo_histogram('norm', batchsize=5000)

depth = dynamic_histograms.equidepth_histogram.Equidepth_Histogram('dynamic_histograms/data/distributions.csv', 500)
depth.create_histogram('norm', l=0, batchsize=5000)

maxdiff = dynamic_histograms.maxdiff_histogram.MaxDiff_Histogram('dynamic_histograms/data/distributions.csv', 500)
maxdiff.create_histogram('norm', batchsize=5000)

sf = dynamic_histograms.sf_histogram.SF_Histogram('dynamic_histograms/data/distributions.csv', 500)
sf.create_histogram('norm', alpha=0.5, m=0.01, s=0.1, batchsize=5000)

spline = dynamic_histograms.spline_histogram.Spline_Histogram('dynamic_histograms/data/distributions.csv', 500)
spline.create_histogram('norm', batchsize=5000)
