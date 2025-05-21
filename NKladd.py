import os
import numpy as np
import matplotlib.pyplot as plt
import dataSet as DS
import Nkod as na
<<<<<<< Updated upstream
import Fkod as fk
=======
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
>>>>>>> Stashed changes

from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
'''cdFull, cdLabels, cdImagesMatrix, cdImagesList = ds.catdog()
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
plt.show()'''
def select_by_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def FTestFeatureSelection(data, n_features=100,return_indices=False):
    """
    Perform F-test feature selection and return filtered dataset with labels.
    
    Parameters:
        data (numpy.ndarray): Dataset with labels in the first column.
        n_features (int): Number of top features to select based on F-statistic.
    
    Returns:
        filtered_data (numpy.ndarray): Dataset with labels and selected features.
    """
    images = data[:, 1:]  # Features
    labels = data[:, 0]   # Labels

    f_values, _ = f_classif(images, labels)
    # Get indices of the top n features based on F-statistic
    top_features = np.argsort(f_values)[-n_features:]  # Select top n features
    top_features = np.sort(top_features)  # Sort indices to maintain column order

    # Filter the features and add labels back as the first column
    filtered_features = images[:, top_features]
    filtered_data = np.column_stack((labels, filtered_features))
    if return_indices:
        return filtered_data, top_features
    return filtered_data


#To do:
#Structure: 
use_catdog = True  # Set to False to use MNIST

if use_catdog:
    full, _, _, _ = DS.catdog()
else:
    full, _, _, _ = DS.mnist()

# Step 1: F-test feature selection
filtered = FTestFeatureSelection(full, n_features=100)
X = filtered[:, 1:]  # exclude label

# Step 2: PCA to 2D
X_pca, pca = select_by_pca(X, n_components=2)

# Optional: visualize
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=filtered[:, 0], cmap='tab10', s=10)
plt.title("PCA after F-test feature selection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Label")
plt.grid(True)
plt.show()