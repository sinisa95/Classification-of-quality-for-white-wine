import math
import numpy as np

#kreiranje recnika na osnovu jednog atributa
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector[0:len(vector)-1])
    return separated

#pravi listu parova
def summarize(dataset):
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
    return summaries

#pravi recnik - key = ocena , value =srednja vr. i st.dev
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    priorClass = {}
    for classValue in separated:
        priorClass[classValue] = len(separated[classValue])/float(len(dataset))

    summaries = {}
    for key in separated:
        summaries[key] = summarize(separated[key])
    return summaries,priorClass


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1/  math.sqrt( (2 * math.pi) * math.pow(stdev,2) ) ) *exponent


def calculateClassProbabilities(summaries, inputVector,priorClass):
    probabilities = {}
    for key in summaries:
        probabilities[key] = priorClass[key]
        for i in range(len(summaries[key])):
            mean, stdev = summaries[key][i]
            x = inputVector[i]
            probabilities[key] *=(calculateProbability(x, mean, stdev))
    return probabilities


def predict(summaries, inputVector, priorClass):
    probabilities = calculateClassProbabilities(summaries, inputVector, priorClass)
    bestLabel, bestProb = None, -1
    for key in probabilities:
        if bestLabel is None or probabilities[key]>bestProb:
            bestProb = probabilities[key]
            bestLabel = key
    return bestLabel


def getPredictions(summaries, testSet,priorClass):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i],priorClass)
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def naiveBayesCall(trainingSet, testSet):
    summaries,priorClass = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet,priorClass)
    accuracy = getAccuracy(testSet, predictions)
    print(accuracy)