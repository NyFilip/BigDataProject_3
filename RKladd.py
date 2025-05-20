import numpy as np
import matplotlib.pyplot as plt
import dataSet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
catsanddogs = dataSet.catdog('catdogdata.txt')[0]
labels_true = catsanddogs[0, :]
images = catsanddogs[1:, :].T
"""
numbers_data = dataSet.mnist('Numbers.txt')[0]
labels_true = numbers_data[0, :]
images = numbers_data[1:, :].T
"""

def perform_dbscan(data, eps, min_samples=5):
    data_scaled = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
    return db.labels_

def plot_clusters(data, labels, title):
    plt.figure(figsize=(6, 4))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(data[mask, 0], data[mask, 1], label=f'Cluster {label}' if label != -1 else 'Noise', s=10)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def perform_pca(data, n_components=30):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

# Perform PCA with 30 components on raw images
images_reduced, pca_model = perform_pca(images, n_components=30)

# Run DBSCAN on the PCA-reduced data
predicted_labels = perform_dbscan(images_reduced, eps=1.5, min_samples=5)

# Further reduce to 2D for visualization
images_2d, _ = perform_pca(images_reduced, n_components=2)

# Plot clusters
plot_clusters(images_2d, predicted_labels, "DBSCAN Clusters (PCA preprocessed)")