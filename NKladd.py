import os
import numpy as np
import matplotlib.pyplot as plt
import dataSet as ds
import Nkod as na
import Fkod as fk

cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()

#mnistLabels = len(set(mLabels))
#catdogLabels = len(set(cdLabels))
#print(f'\n Cat Dog')
cdFtest = fk.FTestFeatureSelection(cdImagesList, 70)
#print(cdFtest.shape)
cdpca_X, cdpca = fk.select_by_pca(cdFtest, 2)
#print(cdpca_X.shape)
cdpred, cddist = na.birch(cdpca_X, cdLabels, 0.3)

#print(f'\n MNIST')
mFtest = fk.FTestFeatureSelection(mImagesList, 200)
#print(mFtest.shape)
mpca_X, mpca = fk.select_by_pca(mFtest, 2)
#print(mpca_X.shape)
mpred, mdist = na.birch(mpca_X, mLabels, .2)

#print(cdpred) 
#print(mpred)
plt.show()
