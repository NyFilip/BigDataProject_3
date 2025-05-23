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
n_runs = 5  # Number of runs for error estimation

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

# --- Storage for clustering results (now stores lists of lists) ---
results = {
    alg: {
        'ari': [[] for _ in sample_sizes],
        'silhouette': [[] for _ in sample_sizes],
        'n_clusters': [[] for _ in sample_sizes]
    }
    for alg in ['KMeans', 'DBSCAN', 'Birch']
}

# --- Loop over different sample sizes and runs ---
for run in range(n_runs):
    for i, size in enumerate(sample_sizes):
        # Balanced sampling: equal from each cluster
        idx = []
        for c in range(n_clusters):
            c_idx = np.where(y_true_full == c)[0]
            idx.extend(np.random.choice(c_idx, size // n_clusters, replace=False))
        X = X_full[idx]
        y_true = y_true_full[idx]

        # --- KMeans ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=None)
        y_kmeans = kmeans.fit_predict(X)
        results['KMeans']['ari'][i].append(adjusted_rand_score(y_true, y_kmeans))
        results['KMeans']['silhouette'][i].append(silhouette_score(X, y_kmeans))
        results['KMeans']['n_clusters'][i].append(len(np.unique(y_kmeans)))

        # --- DBSCAN ---
        dbscan = DBSCAN(eps=4.5, min_samples=5)
        y_dbscan = dbscan.fit_predict(X)
        mask = y_dbscan != -1
        if np.unique(y_dbscan[mask]).size > 1:
            sil = silhouette_score(X[mask], y_dbscan[mask])
            ari = adjusted_rand_score(y_true[mask], y_dbscan[mask])
        else:
            sil = np.nan
            ari = np.nan
        results['DBSCAN']['ari'][i].append(ari)
        results['DBSCAN']['silhouette'][i].append(sil)
        results['DBSCAN']['n_clusters'][i].append(len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0))

        # --- Birch ---
        birch = Birch(n_clusters=n_clusters)
        y_birch = birch.fit_predict(X)
        results['Birch']['ari'][i].append(adjusted_rand_score(y_true, y_birch))
        results['Birch']['silhouette'][i].append(silhouette_score(X, y_birch))
        results['Birch']['n_clusters'][i].append(len(np.unique(y_birch)))

# --- Plot clustering results with error bars ---
colors = {'KMeans': 'tab:orange', 'DBSCAN': 'tab:blue', 'Birch': 'tab:green'}
markers = {'KMeans': 's', 'DBSCAN': 'o', 'Birch': '^'}
metrics = ['ari', 'silhouette', 'n_clusters']

for metric in metrics:
    plt.figure(figsize=(7, 5))
    for alg in results:
        vals = np.array(results[alg][metric], dtype=np.float64)  # shape: (len(sample_sizes), n_runs)
        means = np.nanmean(vals, axis=1)
        stds = np.nanstd(vals, axis=1)
        plt.errorbar(
            sample_sizes, means, yerr=stds,
            marker=markers[alg], color=colors[alg], label=alg,
            linewidth=2, markersize=8, capsize=5
        )
        plt.scatter(sample_sizes, means, color=colors[alg], marker=markers[alg], s=80)
    plt.xlabel('Sample Size')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} vs Sample Size (mean ± std)")
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    if metric == 'ari':
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"part_two/{metric}_results.png", bbox_inches="tight", dpi=300)
    plt.close()

# --- DBSCAN k-distance plot for tuning eps (high quality) ---
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
plt.tight_layout()
plt.savefig("part_two/dbscan_k_distance_plot.png", bbox_inches="tight", dpi=300)
plt.close()
