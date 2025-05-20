import numpy as np
import matplotlib.pyplot as plt
import dataSet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
""""
catsanddogs = dataSet.catdog('catdogdata.txt')[0]
labels_true = catsanddogs[0, :]           # True labels (cats/dogs)
images = catsanddogs[1:, :].T             # Shape: (198, 1024)
"""
numbers_data = dataSet.mnist('Numbers.txt')[0]
labels_true = numbers_data[0, :]
images = numbers_data[1:, :].T

# Function to perform DBSCAN clustering
def perform_dbscan(data, eps=0.5, min_samples=5):
    data_scaled = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
    return db.labels_

# Function to plot 2D clusters
def plot_clusters(data, labels, title):
    plt.figure(figsize=(6, 4))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(data[mask, 0], data[mask, 1], label=f'Cluster {label}' if label != -1 else 'Noise')
    plt.title(title)
    plt.legend()
    plt.show()

# Perform DBSCAN
predicted_labels = perform_dbscan(images, eps=6.0, min_samples=5)

# Reduce to 2D for plotting
pca = PCA(n_components=2)
images_2d = pca.fit_transform(images)

# Plot
plot_clusters(images_2d, predicted_labels, "DBSCAN on Dataset (PCA reduced)")

pca = PCA(n_components=2)
images_2d = pca.fit_transform(images)

plot_clusters(images_2d, labels_true, "Ground Truth (Cats & Dogs)")

for eps in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    labels = perform_dbscan(images, eps=eps, min_samples=5)
    print(f"eps={eps} → clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
