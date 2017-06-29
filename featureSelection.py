from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy

def  selectionAttributes(dataset):

    datasetMatrix = numpy.array(dataset)
    attributes = datasetMatrix[:, 0:11]
    qualities = datasetMatrix[:, 11].reshape(datasetMatrix.shape[0],1)
    #nauci feature selection
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(attributes,qualities)
    selectedColumn = fit.transform(attributes)

    finalDataset = numpy.append(selectedColumn,qualities,1)

    return finalDataset.tolist()