import numpy as np
import matplotlib.pyplot as plt
import dataSet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, homogeneity_score

# ==== Load dataset ====
catsanddogs = dataSet.catdog('catdogdata.txt')[0]
labels_true = catsanddogs[0, :]
images = catsanddogs[1:, :].T  # Shape: (198, 1024)

# ==== Standardize and reduce dimensions ====
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images)

# Reduce to 30D for clustering
pca_cluster = PCA(n_components=30)
images_reduced = pca_cluster.fit_transform(images_scaled)

# ==== Agglomerative Clustering ====
agg = AgglomerativeClustering(n_clusters=2)  # Try n_clusters=2 for cats/dogs
agg_labels = agg.fit_predict(images_reduced)

# ==== Reduce to 2D for plotting ====
pca_vis = PCA(n_components=2)
images_2d = pca_vis.fit_transform(images_reduced)

# ==== Plotting Function ====
def plot_clusters(data, labels, title):
    plt.figure(figsize=(6, 4))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(data[mask, 0], data[mask, 1], label=f'Cluster {label}', s=10)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==== Plots ====
plot_clusters(images_2d, agg_labels, "Agglomerative Clustering (PCA-reduced)")
plot_clusters(images_2d, labels_true, "Ground Truth (PCA-reduced)")

# ==== Evaluation ====
ari = adjusted_rand_score(labels_true, agg_labels)
homo = homogeneity_score(labels_true, agg_labels)
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Homogeneity Score:  {homo:.3f}")