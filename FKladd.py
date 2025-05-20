import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import dataSet
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def K_Elbow(images, k_range=range(1,10),random_state=0):


    inertias = []
    
    for k in k_range:
        model = sklearn.cluster.KMeans(n_clusters=k)
        model.fit(images)
        inertias.append(model.inertia_)

    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

# 1. Feature selection using variance threshold
def select_by_variance(X, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X)
    return X_var, selector.get_support(indices=True)

# 2. Feature selection using PCA (dimensionality reduction)
def select_by_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


castanddogs= dataSet.catdog('catdogdata.txt')[0]
labels=castanddogs[0,:]
images=castanddogs[1:,:]

X_var, selected_indices = select_by_variance(images, threshold=0.01)
X_pca, pca_model = select_by_pca(X_var, n_components=5)

# Print selected feature indices and explained variance ratio
print("Selected feature indices (variance):", selected_indices)
print("Explained variance ratio (PCA):", pca_model.explained_variance_ratio_)








K_Elbow(images=X_pca)