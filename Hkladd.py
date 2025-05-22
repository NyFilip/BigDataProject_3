from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import dataSet as DS
import Nkod as N
import Fkod as F
import Rkod as R    
import Hkod as H

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def visCatDog(cdMatrix):
    fig, axs = plt.subplots(4,4)
    for i in range(16):
        axs[int(np.floor((i)/4)),(i)%4].matshow(cdMatrix[np.random.randint(0, high=198)])
    plt.show()

def visMnist(mMatrix):
    fig, axs = plt.subplots(4,4)
    for i in range(16):
        axs[int(np.floor((i)/4)),(i)%4].matshow(mMatrix[np.random.randint(0, high=2000)])
    plt.show()

def select_by_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
_, catdog_labels, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList  # shape: (198, 4096
_, mnist_labels, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()  # shape: (N, 256)

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

_, _, _, fullcatdog = DS.catdog()
_, _, _, fullmnist = DS.mnist()

prediction, distance = N.birch(dataset, classes, thresh = 0.5)

dblabels_, dbcomponents_ = R.perform_dbscan(data, eps, min_samples=5)
labels = F.run_kmeans(X, n_clusters=3, random_state=0)






def clustering_stability(X, k=3, n_trials=20, sample_frac=0.8, random_state=42):
    """
    Estimates clustering stability using resampling and ARI.

    Parameters:
        X : np.ndarray
            Data to cluster.
        k : int
            Number of clusters.
        n_trials : int
            Number of resampling trials.
        sample_frac : float
            Fraction of data to sample each time.
        random_state : int
            Seed for reproducibility.

    Returns:
        stability_scores : list of float
            List of ARI scores between clustering pairs.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    all_labels = []

    # Run clustering multiple times on random subsets
    for _ in range(n_trials):
        indices = np.random.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        subset = X[indices]
        model = KMeans(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(subset)
        all_labels.append((indices, labels))

    # Compare labelings across pairs of trials on their intersection
    scores = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            idx_i, labels_i = all_labels[i]
            idx_j, labels_j = all_labels[j]
            # Find common indices
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 5:
                continue  # too few points to compare

            # Get labels for common points
            map_i = {idx: label for idx, label in zip(idx_i, labels_i)}
            map_j = {idx: label for idx, label in zip(idx_j, labels_j)}
            common_labels_i = [map_i[idx] for idx in common]
            common_labels_j = [map_j[idx] for idx in common]

            ari = adjusted_rand_score(common_labels_i, common_labels_j)
            scores.append(ari)

    return scores
# Already implemented as `clustering_stability_resample(...)` earlier
# You can now call it like:
scores_catdog = clustering_stability_resample(get_catdog_data, catdog_kmeans)
print("CatDog KMeans Mean ARI:", np.mean(scores_catdog))
