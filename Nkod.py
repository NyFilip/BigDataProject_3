from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np
import Fkod as fk
import dataSet as ds


def birch(dataset, classes, thresh = 0.5):

    brc = Birch(threshold = thresh, n_clusters = classes)
    brc.fit(dataset)
    prediction = brc.predict(dataset)
    distance = brc.fit_transform(dataset)
    return prediction, distance

def truePredPlot(dataset, trueLabels, distance, prediction):
    setLength = len(distance[1])    
    fig, ax = plt.subplots(2)
    
    scatter1 = ax[0].scatter(dataset[:setLength,0], distance[:setLength,1], c=trueLabels[:setLength], marker='o')
    legend1 = ax[0].legend(*scatter1.legend_elements())
    ax[0].add_artist(legend1)
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].set_title('Birch Clustering (True Labels)')

    scatter2 = ax[1].scatter(dataset[:setLength,0], distance[:setLength,1], c=prediction[:setLength], marker='o')
    legend2 = ax[1].legend(*scatter2.legend_elements())
    ax[1].add_artist(legend2)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_title('Birch Clustering (Predicted Labels)')    
    return

def catdogBirch():
    cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
    cdFtest = fk.FTestFeatureSelection(cdFull[:,1:], 768)
    cdpca_X, cdpca = fk.select_by_pca(cdFtest, 12)
    cdPred, cdDist = birch(cdpca_X, 2, .1)
    truePredPlot(cdpca_X, cdLabels, cdDist, cdPred)
    return cdFull

def mnistBirch():
    mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()
    mFtest = fk.FTestFeatureSelection(mFull[:,1:], 150)
    mpca_X, mpca = fk.select_by_pca(mFtest, 12)
    mPred, mDist = birch(mpca_X, 9, 1.7)
    truePredPlot(mpca_X, mLabels, mDist, mPred)

    return mFull
