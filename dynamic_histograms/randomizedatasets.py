import random
import codecs

datasets = ['data/bikesharingtimeseries']
#['data/occupancytraining', 'data/bikesharingtimeseries', 'data/zipfdistributions']
datasetattributes = {'data/occupancytraining.csv': ["Humidity","Light","CO2","HumidityRatio"],
                     'data/bikesharingtimeseries.csv': ['casual','registered','cnt'],
                     'data/zipfdistributions.csv': ['zipf1.05','zipf1.1','zipf1.15','zipf1.2','zipf1.25','zipf1.3']}

for dataset in datasets:
    for i in range(10,11):
        headers = None
        lines = None
        with codecs.open(dataset + ".csv", encoding="utf-8") as f:
            headers = f.readline()
            lines = f.readlines()
        linescopy = list(lines)
        random.shuffle(linescopy)
        with codecs.open(dataset + str(i) + ".csv", encoding="utf-8", mode="w") as w:
            w.write(headers)
            w.writelines(linescopy)

