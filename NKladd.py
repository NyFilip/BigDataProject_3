import os
import numpy as np
import matplotlib.pyplot as plt
import dataSet as ds
import Nkod as na
import Fkod as fk

np.set_printoptions(threshold=np.inf)

def catdogBirch():
    cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
    cdFtest = fk.FTestFeatureSelection(cdImagesList, 70)
    cdpca_X, cdpca = fk.select_by_pca(cdFtest, 2)
    cdpred, cddist = na.birch(cdpca_X, cdLabels, 0.3)
    return cdFull

def mnistBirch():
    mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()
    mFtest = fk.FTestFeatureSelection(mImagesList, 200)
    mpca_X, mpca = fk.select_by_pca(mFtest, 2)
    mpred, mdist = na.birch(mpca_X, mLabels, .2)
    return mFull



if __name__ == '__main__':
    cdImagesMatrix = catdogBirch()
    mImagesMatrix = mnistBirch()
    #print(cdImagesMatrix[4])
    #print(mImagesMatrix[6])
    plt.show()
