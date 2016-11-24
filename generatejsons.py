import json

numworkersperhistogram = 5
numhistograms = 7
numdatasets = 5

numworkersperdataset = numhistograms * numworkersperhistogram

# we need somewhere to host all of the bucket json's

# each dataset will be a different HIT, with each HIT showing either control, dc, dvo, 
# equidepth, maxdiff, sf, or spline

# since each HIT will be a one dataset, there will be as many different sets of questions 
# as there are different datasets

# the website will need to account for all of these different histograms and questions
# the name of the dataset will determine which questions to display and which set of histograms to 
# choose from 

# we will have two dictionaries

# first dictionary will be (k, v), with k being the dataset and v being a list of questions for each dataset

# second dictionary will be (k, v) with k being the dataset and v being the list of histograms 

# can implement barebones and then email Emanuel to ask about design later today

# should be a list of dictionaries for the json, each json object is a dictionary and this dictionary can 
# contain a question and the answers

# histogramlist https://api.myjson.com/bins/3z96a

datasets = ["bikesharing", "occupancy", "zipf1", "zipf2", "zipf3"]

bikesharingcnt50 = ["https://api.myjson.com/bins/3q93m", "https://api.myjson.com/bins/28o36", 
"https://api.myjson.com/bins/1risy", "https://api.myjson.com/bins/n0gy", "https://api.myjson.com/bins/49r9e", 
"https://api.myjson.com/bins/2lqrm", "https://api.myjson.com/bins/22gbm"]
# control, dc, dvo, equidepth, maxdiff, sf, spline

occupancyCO250 = ["https://cs.brown.edu/~ez/histogram/CO2control50.json",
				  "https://cs.brown.edu/~ez/histogram/CO2dc50.json", "https://cs.brown.edu/~ez/histogram/CO2dvo50.json",
				  "https://cs.brown.edu/~ez/histogram/CO2equidepth50.json",
				  "https://cs.brown.edu/~ez/histogram/CO2maxdiff50.json", "https://cs.brown.edu/~ez/histogram/CO2sf50.json",
				  "https://cs.brown.edu/~ez/histogram/CO2spline50.json"]

histogramlist = {}
questionlist = {}

for dataset in datasets:
	if dataset == "bikesharing":
		histogramlist[dataset] = [{"control": bikesharingcnt50[0]}, {'dc': bikesharingcnt50[1]}, 
			{'dvo': bikesharingcnt50[2]}, {'equidepth': bikesharingcnt50[3]}, {'maxdiff': bikesharingcnt50[4]}, 
			{'sf': bikesharingcnt50[5]}, {'spline': bikesharingcnt50[6]}]
		questionlist[dataset] = [
		{"Q1": "What is the frequency of 250? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["~1,000", "~2,500", "~3,000","I don't know"],
		"dc": ["~3,000", "~300,000", "~1,250","I don't know"],
		"dvo": ["~1,000", "~2,750", "~600","I don't know"],
		"equidepth": ["~0", "~1,000", "~600","I don't know"],
		"maxdiff": ["~0", "~6,000", "~400","I don't know"],
		"sf": ["~0", "~2,700", "~600","I don't know"],
		"spline": ["~2,500", "~0", "~500", "I don't know"]},
		{"Q2": "Which histogram buckets hold values whose frequencies are greater then 2,000? You may choose multiple answers." + 
		"You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
		"dc": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
		"dvo": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
		"equidepth": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
		"maxdiff": ["#1", "#2", "#3", "#5", "#8", "#I don't know"],
		"sf": ["#1", "#2", "#3", "#5", "#8", "#I don't know"],
		"spline": ["#1", "#2", "#3", "#5", "#8", "I don't know"]},
		{"Q3": "How many values are in the range 98.7 - 196.4? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["~3,250", "~325,000", "~7,000", "I don't know"],
		"dc": ["~4,250", "~425,000", "~7,000", "I don't know"],
		"dvo": ["~3,750", "~~375,000", "~7,000", "I don't know"],
		"equidepth": ["~16,000", "~10,000", "~2,000", "I don't know"],
		"maxdiff": ["~11,000", "~6,000", "~600,000", "I don't know"],
		"sf": ["~4,000", "~7,000", "~400,000", "I don't know"],
		"spline": ["~3,500", "~350,000", "~16,000", "I don't know"]},
		{"Q4": "What is the maximum value of the histogram? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["~980", "~12,000", "~700", "I don't know"],
		"dc": ["~980", "~12,000", "~700", "I don't know"],
		"dvo": ["~980", "~12,000", "~700", "I don't know"],
		"equidepth": ["~980", "~12,000", "~700", "I don't know"],
		"maxdiff": ["~980", "~12,000", "~700", "I don't know"],
		"sf": ["~980", "~12,000", "~700", "I don't know"],
		"spline": ["~980", "~12,000", "~700", "I don't know"]},
		{"Q5": "What is the minimum value of the histogram? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"],
		"dc": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"],
		"dvo": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"],
		"equidepth": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"],
		"maxdiff": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"],
		"sf": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"],
		"spline": ["~0 - ~12,500", "~0 - ~1,000", "~500 - ~12,500", "I don't know"]},
		{"Q6": "What is the distribution of the data? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
		"dc": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
		"dvo": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
		"equidepth": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
		"maxdiff": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
		"sf": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
		"spline": ["Normal", "Right Skewed", "Left Skewed", "I don't know"]},
		{"Q7": "Are there any outliers in the data? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["Yes", "No", "I don't know"],
		"dc": ["Yes", "No", "I don't know"],
		"dvo": ["Yes", "No", "I don't know"],
		"equidepth": ["Yes", "No", "I don't know"],
		"maxdiff": ["Yes", "No", "I don't know"],
		"sf": ["Yes", "No", "I don't know"],
		"spline": ["Yes", "No", "I don't know"]},
		{"Q8": "Are there histogram buckets with similar frequencies? You may move your cursor over the histogram to see the range of each bucket.",
		"control": ["Yes", "No", "I don't know"],
		"dc": ["Yes", "No", "I don't know"],
		"dvo": ["Yes", "No", "I don't know"],
		"equidepth": ["Yes", "No", "I don't know"],
		"maxdiff": ["Yes", "No", "I don't know"],
		"sf": ["Yes", "No", "I don't know"],
		"spline": ["Yes", "No", "I don't know"]}]
	elif dataset == "occupancy":
		histogramlist[dataset] = [{"control": occupancyCO250[0]}, {'dc': occupancyCO250[1]},
								  {'dvo': occupancyCO250[2]}, {'equidepth': occupancyCO250[3]},
								  {'maxdiff': occupancyCO250[4]},
								  {'sf': occupancyCO250[5]}, {'spline': occupancyCO250[6]}]
		questionlist[dataset] = [
			{"Q1": "What is the frequency of 100? You may move your cursor over the histogram to see the range of each bucket.",
			 "control": ["~13,000", "~12,000", "~50,000", "I don't know"],
			 "dc": ["~6,250", "~15,000", "~1,000", "I don't know"],
			 "dvo": ["~13,000", "~11,000", "~50,000", "I don't know"],
			 "equidepth": ["~1,750", "~1,000", "~5,000", "I don't know"],
			 "maxdiff": ["~1,000", "~15,000", "~5,000", "I don't know"],
			 "sf": ["~12,250", "~20,000", "~6,000", "I don't know"],
			 "spline": ["~1,000", "~15,000", "~5,000", "I don't know"]},
			{"Q2": "Which histogram buckets hold values whose frequencies are greater then 2,000? You may choose multiple answers." +
			"You may move your cursor over the histogram to see the range of each bucket.",
			"control": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
			"dc": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
			"dvo": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
			"equidepth": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
			"maxdiff": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
			"sf": ["#1", "#2", "#3", "#5", "#8", "I don't know"],
			"spline": ["#1", "#2", "#3", "#5", "#8", "I don't know"]},
			{"Q3": "How many values are in the range 158.2 - 316.4? You may move your cursor over the histogram to see the range of each bucket.",
			 "control": ["~4,000", "~500", "~11,000", "I don't know"],
			 "dc": ["~6,250", "~25,000", "~2,000", "I don't know"],
			 "dvo": ["~1,750", "~~5,000", "~12,000", "I don't know"],
			 "equidepth": ["~100", "~10,000", "~1,000", "I don't know"],
			 "maxdiff": ["~1,000", "~6,000", "~20,000", "I don't know"],
			 "sf": ["~2,000", "~7,000", "~50,000", "I don't know"],
			 "spline": ["~1,000", "~6,000", "~20,000", "I don't know"]},
			{"Q4": "What is the maximum value of the histogram? You may move your cursor over the histogram to see the range of each bucket.",
			 "control": ["~15,900", "~20,000", "~10,000", "I don't know"],
			 "dc": ["~15,900", "~20,000", "~10,000", "I don't know"],
			 "dvo": ["~15,900", "~20,000", "~10,000", "I don't know"],
			 "equidepth": ["~15,900", "~20,000", "~10,000", "I don't know"],
			 "maxdiff": ["~15,900", "~20,000", "~10,000", "I don't know"],
			 "sf": ["~15,900", "~20,000", "~10,000", "I don't know"],
			 "spline": ["~15,900", "~20,000", "~10,000", "I don't know"]},
			{"Q5": "What is the minimum value of the histogram? You may move your cursor over the histogram to see the range of each bucket.",
			 "control": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"],
			 "dc": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"],
			 "dvo": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"],
			 "equidepth": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"],
			 "maxdiff": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"],
			 "sf": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"],
			 "spline": ["~0 - ~15,900", "~0 - ~16,000", "~0 - ~14,000", "I don't know"]},
			{"Q6": "What is the distribution of the data You may move your cursor over the histogram to see the range of each bucket.?",
			 "control": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
			 "dc": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
			 "dvo": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
			 "equidepth": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
			 "maxdiff": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
			 "sf": ["Normal", "Right Skewed", "Left Skewed", "I don't know"],
			 "spline": ["Normal", "Right Skewed", "Left Skewed", "I don't know"]},
			{"Q7": "Are there any outliers in the data? You may move your cursor over the histogram to see the range of each bucket.",
			 "control": ["Yes", "No", "I don't know"],
			 "dc": ["Yes", "No", "I don't know"],
			 "dvo": ["Yes", "No", "I don't know"],
			 "equidepth": ["Yes", "No", "I don't know"],
			 "maxdiff": ["Yes", "No", "I don't know"],
			 "sf": ["Yes", "No", "I don't know"],
			 "spline": ["Yes", "No", "I don't know"]},
			{"Q8": "Are there histogram buckets with similar frequencies? You may move your cursor over the histogram to see the range of each bucket.",
			 "control": ["Yes", "No", "I don't know"],
			 "dc": ["Yes", "No", "I don't know"],
			 "dvo": ["Yes", "No", "I don't know"],
			 "equidepth": ["Yes", "No", "I don't know"],
			 "maxdiff": ["Yes", "No", "I don't know"],
			 "sf": ["Yes", "No", "I don't know"],
			 "spline": ["Yes", "No", "I don't know"]}]
	else:
		histogramlist[dataset] = []

with open("datasetlinks.json", 'w') as f:
	json.dump(histogramlist, f)

with open("histogramquestions.json", "w") as q:
	json.dump(questionlist, q)