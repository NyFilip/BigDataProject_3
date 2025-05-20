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

pca_cluster = PCA(n_components=30)
images_reduced = pca_cluster.fit_transform(images)

predicted_labels = perform_dbscan(images_reduced, eps=1.5, min_samples=5)

pca_vis = PCA(n_components=2)
images_2d = pca_vis.fit_transform(images_reduced)

plot_clusters(images_2d, predicted_labels, "DBSCAN Clusters (PCA preprocessed)")

plot_clusters(images_2d, labels_true, "Ground Truth (PCA projection)")

print("Tuning eps for PCA-reduced data:")
for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    labels = perform_dbscan(images_reduced, eps=eps, min_samples=5)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"eps={eps} → clusters found: {n_clusters}")
