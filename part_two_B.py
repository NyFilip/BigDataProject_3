import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import os

# --- Parameters ---
n_clusters = 5
n_features = 10
n_samples = 3000
imbalance_ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Ratio of largest to smallest cluster

# --- Storage for results ---
results = {alg: {'ari': [], 'silhouette': [], 'n_clusters': []} for alg in ['KMeans', 'DBSCAN', 'Birch']}

for ratio in imbalance_ratios:
    # --- Generate imbalanced cluster sizes ---
    # Smallest cluster size
    min_size = n_samples // (sum(ratio**i for i in range(n_clusters)))
    sizes = [min_size * (ratio**i) for i in range(n_clusters)]
    sizes[-1] += n_samples - sum(sizes)  # Adjust last cluster to match total

    centers = np.random.uniform(-20, 20, size=(n_clusters, n_features))
    X, y_true = make_blobs(
        n_samples=sizes,
        centers=centers,
        n_features=n_features,
        cluster_std=1.0,
        random_state=42
    )

    # --- KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    results['KMeans']['ari'].append(adjusted_rand_score(y_true, y_kmeans))
    results['KMeans']['silhouette'].append(silhouette_score(X, y_kmeans))
    results['KMeans']['n_clusters'].append(len(np.unique(y_kmeans)))

    # --- DBSCAN ---
    dbscan = DBSCAN(eps=1.8, min_samples=5)  # You may need to tune eps
    y_dbscan = dbscan.fit_predict(X)
    mask = y_dbscan != -1
    if np.unique(y_dbscan[mask]).size > 1:
        sil = silhouette_score(X[mask], y_dbscan[mask])
    else:
        sil = np.nan
    results['DBSCAN']['ari'].append(adjusted_rand_score(y_true[mask], y_dbscan[mask]) if mask.any() else np.nan)
    results['DBSCAN']['silhouette'].append(sil)
    results['DBSCAN']['n_clusters'].append(len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0))

    # --- Birch ---
    birch = Birch(n_clusters=n_clusters)
    y_birch = birch.fit_predict(X)
    results['Birch']['ari'].append(adjusted_rand_score(y_true, y_birch))
    results['Birch']['silhouette'].append(silhouette_score(X, y_birch))
    results['Birch']['n_clusters'].append(len(np.unique(y_birch)))

# --- Plotting ---
plt.figure(figsize=(15, 5))
colors = {'KMeans': 'tab:orange', 'DBSCAN': 'tab:blue', 'Birch': 'tab:green'}
markers = {'KMeans': 's', 'DBSCAN': 'o', 'Birch': '^'}
xlabels = [f"1:{r}" for r in imbalance_ratios]

for i, metric in enumerate(['ari', 'silhouette', 'n_clusters']):
    plt.subplot(1, 3, i+1)
    for alg in results:
        y = np.array(results[alg][metric], dtype=np.float64)
        plt.plot(
            xlabels, y,
            marker=markers[alg], color=colors[alg], label=alg, linewidth=2, markersize=8
        )
        plt.scatter(xlabels, y, color=colors[alg], marker=markers[alg], s=80)
    plt.xlabel('Cluster Size Ratio (smallest:largest)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(metric.replace('_', ' ').title())
    plt.grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])

os.makedirs("part_two", exist_ok=True)
plt.savefig("part_two/cluster_imbalance_results.png", bbox_inches="tight")
plt.show()