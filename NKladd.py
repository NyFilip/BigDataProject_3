import os
import numpy as np
import matplotlib.pyplot as plt
import dataSet as ds
import Nkod as na

cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()

#mnistLabels = len(set(mLabels))
#catdogLabels = len(set(cdLabels))

cdpred, cddist = na.birch(cdImagesList, cdLabels)
mpred, mdist = na.birch(mImagesList, mLabels)

#print(cdpred) 
#print(mpred)
plt.show()
