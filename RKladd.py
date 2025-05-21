import numpy as np
import matplotlib.pyplot as plt
import dataSet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from sklearn.neighbors import NearestNeighbors

def plot_k_distance(data, k=5):
    """
    Plot k-distance (distance to k-th nearest neighbor) to help choose DBSCAN eps.
    """
    data_scaled = StandardScaler().fit_transform(data)
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)
    k_distances = np.sort(distances[:, k-1])  # distance to k-th neighbor

    plt.figure(figsize=(6, 4))
    plt.plot(k_distances)
    plt.title(f"{k}-distance Plot for DBSCAN")
    plt.xlabel("Sorted Points")
    plt.ylabel(f"Distance to {k}-th nearest neighbor")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def test_stability(data, cluster_fn, labels_ref=None, n_iter=100, sample_frac=0.8, random_state=None, title="Clustering Stability"):
    
    # Minimal resampling stability test for any clustering function.
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    if labels_ref is None:
        labels_ref = cluster_fn(data)

    ari_scores = []
    for _ in range(n_iter):
        idx = np.random.choice(n, size=int(n * sample_frac), replace=True)
        data_sample = data[idx]
        labels_sample = cluster_fn(data_sample)
        ari = adjusted_rand_score(labels_ref[idx], labels_sample) if len(set(labels_sample)) > 1 else 0
        ari_scores.append(ari)

    plt.hist(ari_scores, bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel("ARI Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print(f"Mean ARI: {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")


# Perform PCA with 30 components on raw images
images_reduced, pca_model = perform_pca(images, n_components=30)

plot_k_distance(images_reduced, k=5)  # or min_samples

# Run DBSCAN on the PCA-reduced data
predicted_labels = perform_dbscan(images_reduced, eps=2.8, min_samples=5)

# Further reduce to 2D for visualization
#images_2d, _ = perform_pca(images_reduced, n_components=2)

# Plot clusters
#plot_clusters(images_2d, predicted_labels, "DBSCAN Clusters (PCA preprocessed)")

# Run stability test for DBSCAN
dbscan_fn = lambda X: perform_dbscan(X, eps=2.8, min_samples=5)
test_stability(images_reduced, dbscan_fn, labels_ref=predicted_labels, title="DBSCAN Stability (ARI)")