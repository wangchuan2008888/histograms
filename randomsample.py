import csv
import random
import sys

file = 'dynamic_histograms/data/USCensus1990.csv'
attr = 'dRpincome'
output = 'dynamic_histograms/data/USCensusThirtySamples.csv'

def calculateSkip(n):
    v = random.uniform(0, 1)
    l = 0
    t = n + 1
    num = 1
    quot = num / t
    while quot > v:
        l += 1
        t += 1
        num += 1
        quot = (quot * num) / t
    return l

def maintainBackingSample(value, sample):
    #if len(sample) + 1 <= self.upper:
    #    sample.append(value)
    #else:
    rand_index = random.randint(0, len(sample) - 1)
    sample[rand_index] = value
    #if self.min not in sample:
    #    sample.append(self.min)
    #if self.max not in sample:
    #    sample.append(self.max)
    return sample

with open(file) as f:
    reader = csv.reader(f)
    header = reader.next()
    for i in range(0, len(header)):
        header[i] = unicode(header[i], 'utf-8-sig')
    #print header
    #sys.exit(0)
    #attr_index = header.index(attr)
    samplenum = 30
    samplesize = 10000
    backingsamples = []
    initial = True
    skips = {}
    skipscounter = {}
    N = 0
    for i in range(samplenum):
        backingsamples.append([])
    for row in reader:
        N += 1
        if initial:
            for i in range(samplenum):
                #backingsamples[i].append(float(row[attr_index]))
                backingsamples[i].append(row)
        if len(backingsamples[0]) == samplesize and initial:
            initial = False
            for i in range(samplenum):
                skips[str(i)] = calculateSkip(N)
                skipscounter[str(i)] = 0
        elif initial == False:
            for i in range(samplenum):
                skipscounter[str(i)] += 1
                if skips[str(i)] == skipscounter[str(i)]:
                    #backingsamples[i] = maintainBackingSample(float(row[attr_index]), backingsamples[i])
                    backingsamples[i] = maintainBackingSample(row, backingsamples[i])
                    skipscounter[str(i)] = 0
                    skips[str(i)] = calculateSkip(N)
        if N % 100000 == 0:
            print N

with open(output, 'wb') as o:
    writer = csv.writer(o)
    #writer.writerow([attr])
    writer.writerow(header)
    for i in range(samplenum):
        print i
        for j in range(len(backingsamples[i])):
            writer.writerow(backingsamples[i][j])