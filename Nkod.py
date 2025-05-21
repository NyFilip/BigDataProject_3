from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np
import Fkod as fk
import dataSet as ds


def birch(dataset, labels, thresh = 0.5):
    clusters = len(set(labels))
    
    brc = Birch(threshold = thresh, n_clusters = clusters)
    brc.fit(dataset)
    prediction = brc.predict(dataset)
    distance = brc.fit_transform(dataset)
    #print(clusters)
    #print(f'predictions:{prediction.shape}, distance:{distance.shape}')
    #print(dataset.shape)
    setLength = len(distance[1])
    
    plt.figure()
    plt.scatter(dataset[:setLength,0], distance[:setLength,1], c=prediction[:setLength], marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Birch Clustering')
    
    return prediction, distance

def catdogBirch():
    cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
    cdFtest = fk.FTestFeatureSelection(cdImagesList, 70)
    cdpca_X, cdpca = fk.select_by_pca(cdFtest, 2)
    cdpred, cddist = birch(cdpca_X, cdLabels, 0.3)
    return cdFull

def mnistBirch():
    mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()
    mFtest = fk.FTestFeatureSelection(mImagesList, 250)
    mpca_X, mpca = fk.select_by_pca(mFtest, 2)
    mpred, mdist = birch(mpca_X, mLabels, 0.6)
    return mFull
