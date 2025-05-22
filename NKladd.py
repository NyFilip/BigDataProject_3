import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import dataSet as ds
import Nkod as na


np.set_printoptions(threshold=np.inf)




if __name__ == '__main__':
    na.catdogBirch()
    na.mnistBirch()

    plt.show()
