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
from sklearn.metrics import adjusted_rand_score

_, catdog_labels, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList  # shape: (198, 4096
_, mnist_labels, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()  # shape: (N, 256)

def clustering_stability_Birch(X, k=3,thresh=0.5, n_trials=20, sample_frac=0.8, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    all_labels = []

    for _ in range(n_trials):
        indices = np.random.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        subset = X[indices]
        labels, _ = N.birch(subset, k, thresh)
        all_labels.append((indices, labels))

    scores = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            idx_i, labels_i = all_labels[i]
            idx_j, labels_j = all_labels[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 5:
                continue
            map_i = {idx: label for idx, label in zip(idx_i, labels_i)}
            map_j = {idx: label for idx, label in zip(idx_j, labels_j)}
            common_labels_i = [map_i[idx] for idx in common]
            common_labels_j = [map_j[idx] for idx in common]
            scores.append(adjusted_rand_score(common_labels_i, common_labels_j))

    return scores


def clustering_stability_DB(X, eps, min_samples=5, n_trials=20, sample_frac=0.8, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    all_labels = []

    for _ in range(n_trials):
        indices = np.random.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        subset = X[indices]
        labels, _ = R.perform_dbscan(subset, eps, min_samples)
        all_labels.append((indices, labels))

    scores = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            idx_i, labels_i = all_labels[i]
            idx_j, labels_j = all_labels[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 5:
                continue
            map_i = {idx: label for idx, label in zip(idx_i, labels_i)}
            map_j = {idx: label for idx, label in zip(idx_j, labels_j)}
            common_labels_i = [map_i[idx] for idx in common]
            common_labels_j = [map_j[idx] for idx in common]
            scores.append(adjusted_rand_score(common_labels_i, common_labels_j))

    return scores

def clustering_stability_Kmeans(X, k=3, n_trials=20, sample_frac=0.8, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    all_labels = []

    for _ in range(n_trials):
        indices = np.random.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        subset = X[indices]
        labels = F.run_kmeans(subset, k, random_state=random_state)
        all_labels.append((indices, labels))

    scores = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            idx_i, labels_i = all_labels[i]
            idx_j, labels_j = all_labels[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 5:
                continue
            map_i = {idx: label for idx, label in zip(idx_i, labels_i)}
            map_j = {idx: label for idx, label in zip(idx_j, labels_j)}
            common_labels_i = [map_i[idx] for idx in common]
            common_labels_j = [map_j[idx] for idx in common]
            scores.append(adjusted_rand_score(common_labels_i, common_labels_j))

    return scores
scores_k = clustering_stability_Kmeans(X_mnist, k=9)
scores_b = clustering_stability_Birch(X_mnist, k=9,thresh=1.7,n_trials=20, sample_frac=0.8, random_state=42)
scores_d = clustering_stability_DB(X_mnist, eps = 3.2, min_samples=2, n_trials=20, sample_frac=0.8, random_state=42)

print("KMeans Stability:", np.mean(scores_k))
print("Birch Stability:", np.mean(scores_b))
print("DBSCAN Stability:", np.mean(scores_d))

scores_k = clustering_stability_Kmeans(X_catdog,k=2)
scores_b = clustering_stability_Birch(X_catdog, k=2,thresh=0.1,n_trials=20, sample_frac=0.8, random_state=42)
scores_d = clustering_stability_DB(X_catdog, eps = 3.5, min_samples=4, n_trials=20, sample_frac=0.8, random_state=42)

print("KMeans Stability:", np.mean(scores_k))
print("Birch Stability:", np.mean(scores_b))
print("DBSCAN Stability:", np.mean(scores_d))