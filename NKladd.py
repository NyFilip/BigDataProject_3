import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import dataSet as ds
import Nkod as na
import Rkod as re


np.set_printoptions(threshold=np.inf)




if __name__ == '__main__':
    mFull, mLabels, mImagesMatrix, mImagesList = ds.mnist()
    cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()

    cdPred, cdDist = na.catdogBirch()
    mPred, mDist = na.mnistBirch()
    
    mSil = re.evaluate_clustering(mFull[:,1:], mPred, mLabels)
    cdSil = re.evaluate_clustering(cdFull[:,1:], cdPred, cdLabels)
    na.truePredPlot(mFull[:,1:], mLabels, mDist, mPred)
    print(f'{mSil=}')
    print(f'{cdSil=}')

    plt.show()
