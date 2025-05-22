import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# --- Ensure output directory exists ---
os.makedirs("part_two", exist_ok=True)

# --- Parameters ---
n_clusters = 10
n_features = 50  # High dimensional like MNIST
n_samples_full = 5000
sample_sizes = [5000, 2500, 1000, 500, 250, 100, 50]

# --- Generate synthetic MNIST-like data ---
X_full, y_true_full = make_blobs(
    n_samples=n_samples_full,
    centers=n_clusters,
    n_features=n_features,
    cluster_std=0.5,
    random_state=42
)

# --- PCA Visualization of full data ---
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_full)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_true_full, cmap='tab10', s=10, alpha=0.7)
plt.title("PCA Projection of Synthetic MNIST-like Data")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar(scatter, label="True Cluster")
plt.tight_layout()
plt.savefig("part_two/MNIST.png", bbox_inches="tight")
plt.show()

# --- Storage for clustering results ---
results = {alg: {'ari': [], 'silhouette': [], 'n_clusters': []} for alg in ['KMeans', 'DBSCAN', 'Birch']}

# --- Loop over different sample sizes ---
for size in sample_sizes:
    # Balanced sampling: equal from each cluster
    idx = []
    for c in range(n_clusters):
        c_idx = np.where(y_true_full == c)[0]
        idx.extend(np.random.choice(c_idx, size // n_clusters, replace=False))
    X = X_full[idx]
    y_true = y_true_full[idx]

    # --- KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    results['KMeans']['ari'].append(adjusted_rand_score(y_true, y_kmeans))
    results['KMeans']['silhouette'].append(silhouette_score(X, y_kmeans))
    results['KMeans']['n_clusters'].append(len(np.unique(y_kmeans)))

    # --- DBSCAN ---
    dbscan = DBSCAN(eps=4.5, min_samples=5)  # eps may need tuning
    y_dbscan = dbscan.fit_predict(X)
    mask = y_dbscan != -1
    if np.unique(y_dbscan[mask]).size > 1:
        sil = silhouette_score(X[mask], y_dbscan[mask])
        ari = adjusted_rand_score(y_true[mask], y_dbscan[mask])
    else:
        sil = np.nan
        ari = np.nan
    results['DBSCAN']['ari'].append(ari)
    results['DBSCAN']['silhouette'].append(sil)
    results['DBSCAN']['n_clusters'].append(len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0))

    # --- Birch ---
    birch = Birch(n_clusters=n_clusters)
    y_birch = birch.fit_predict(X)
    results['Birch']['ari'].append(adjusted_rand_score(y_true, y_birch))
    results['Birch']['silhouette'].append(silhouette_score(X, y_birch))
    results['Birch']['n_clusters'].append(len(np.unique(y_birch)))

# --- Plot clustering results across sample sizes ---
plt.figure(figsize=(15, 5))
colors = {'KMeans': 'tab:orange', 'DBSCAN': 'tab:blue', 'Birch': 'tab:green'}
markers = {'KMeans': 's', 'DBSCAN': 'o', 'Birch': '^'}

for i, metric in enumerate(['ari', 'silhouette', 'n_clusters']):
    plt.subplot(1, 3, i + 1)
    for alg in results:
        y = results[alg][metric]
        plt.plot(
            sample_sizes, y,
            marker=markers[alg], color=colors[alg], label=alg, linewidth=2, markersize=8
        )
        plt.scatter(sample_sizes, y, color=colors[alg], marker=markers[alg], s=80)
    plt.xlabel('Sample Size')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(metric.replace('_', ' ').title())
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("part_two/clustering_results.png", bbox_inches="tight")
plt.show()

# --- DBSCAN k-distance plot for tuning eps ---
k = 5  # min_samples for DBSCAN
nbrs = NearestNeighbors(n_neighbors=k).fit(X_full)
distances, indices = nbrs.kneighbors(X_full)
k_distances = np.sort(distances[:, k - 1])

plt.figure(figsize=(6, 4))
plt.plot(k_distances)
plt.ylabel(f"{k}-NN distance")
plt.xlabel("Points sorted by distance")
plt.title("DBSCAN k-distance plot (use elbow as eps)")
plt.grid(True)
plt.savefig("part_two/dbscan_k_distance_plot.png", bbox_inches="tight")
plt.show()
