import datasetFunctions as df
from naiveBayes import  naiveBayesCall
from featureSelection import selectionAttributes
from neuralNetwork import  NeuralNetwork
from neuralNetwork import normalizeData


def main():
    splitRatio = 0.8
    dataset = df.loadCsv(r"dataset.csv")

    groupedDataset = df.groupDatasetByQuality(dataset)

    workingDataset = selectionAttributes(groupedDataset)
    workingDataset = normalizeData(workingDataset)
    trainingSet, testSet = df.splitDataset(workingDataset, splitRatio)
    naiveBayesCall(trainingSet, testSet)
    NeuralNetwork(trainingSet, testSet)

main()









