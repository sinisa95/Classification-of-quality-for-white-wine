import csv
import random

#ucitavanje csv file tj data-seta u memoriju
def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

#deljenje dataset-a za obucavanje i za testiranje
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def groupDatasetByQuality(dataset):
    for i in range (len(dataset)):
        if(dataset[i][11] <=3):
            dataset[i][11]=0
        elif(3<dataset[i][11]<8):
            dataset[i][11] = 1
        else:
            dataset[i][11] = 2

    return dataset