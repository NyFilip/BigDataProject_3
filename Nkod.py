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
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(dataset[:setLength,0], distance[:setLength,1], c=labels[:setLength], marker='o')
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
    ax.set_title('Birch Clustering')
    
    return prediction, distance

def catdogBirch():
    cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
    cdFtest = fk.FTestFeatureSelection(cdFull[:,1:], 768)
    cdpca_X, cdpca = fk.select_by_pca(cdFtest, 12)
    cdpred, cddist = birch(cdpca_X, cdLabels, .1)
    return cdFull

def mnistBirch():
    mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()
    mFtest = fk.FTestFeatureSelection(mFull[:,1:], 150)
    mpca_X, mpca = fk.select_by_pca(mFtest, 12)
    mpred, mdist = birch(mpca_X, mLabels, 1.7)
    return mFull
