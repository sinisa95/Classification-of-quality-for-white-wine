from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import np_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def NeuralNetwork(trainingSet,testSet):
    model = Sequential()
    model.add(Dense(4, activation='relu', input_shape=(4,) ))
    model.add(Dense(8, activation='relu' ))
    model.add(Dense(3, activation='relu' ))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    training = np.array(trainingSet)
    X= training[:,:4]
    Y= training[:,4]
    Y = np_utils.to_categorical(Y, 3)

    model.fit(X,Y, epochs=100 , batch_size=1)
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    test = np.array(testSet)
    testX = test[:, :4]
    testY = test[:, 4]

    predictions = model.predict(testX)
    result = getBest(predictions)
    print(getAccuracy(testY, result))

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i]== predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def getBest(predictions):
    retVal = []
    for x in predictions:
        best = 0
        for i in range(1,len(x)):
            if(x[i] > x[best]):
                best = i
        retVal.append(best)
    return retVal

def normalizeData(dataset):
    i=0
    k=0
    for attribute in zip(*dataset):
        if(i == 4):
            continue
        meanAtr= np.mean(attribute)
        for x in attribute:
            dataset[k][i]/= meanAtr
            k+=1
        k=0
        i+=1
    return dataset

