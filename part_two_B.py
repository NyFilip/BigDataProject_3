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

n_runs = 5  # Number of runs for error estimation

# --- Storage for results (lists of lists for each ratio) ---
results = {
    alg: {
        'ari': [[] for _ in imbalance_ratios],
        'silhouette': [[] for _ in imbalance_ratios],
        'n_clusters': [[] for _ in imbalance_ratios]
    }
    for alg in ['KMeans', 'DBSCAN', 'Birch']
}

for run in range(n_runs):
    for i, ratio in enumerate(imbalance_ratios):
        # --- Generate imbalanced cluster sizes ---
        min_size = n_samples // (sum(ratio**j for j in range(n_clusters)))
        sizes = [min_size * (ratio**j) for j in range(n_clusters)]
        sizes[-1] += n_samples - sum(sizes)  # Adjust last cluster to match total

        centers = np.random.uniform(-20, 20, size=(n_clusters, n_features))
        X, y_true = make_blobs(
            n_samples=sizes,
            centers=centers,
            n_features=n_features,
            cluster_std=1.0,
            random_state=None  # Use different random state each run
        )

        # --- KMeans ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=None)
        y_kmeans = kmeans.fit_predict(X)
        results['KMeans']['ari'][i].append(adjusted_rand_score(y_true, y_kmeans))
        results['KMeans']['silhouette'][i].append(silhouette_score(X, y_kmeans))
        results['KMeans']['n_clusters'][i].append(len(np.unique(y_kmeans)))

        # --- DBSCAN ---
        dbscan = DBSCAN(eps=1.8, min_samples=5)
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

# --- Plotting (mean ± std error bars for each metric) ---
colors = {'KMeans': 'tab:orange', 'DBSCAN': 'tab:blue', 'Birch': 'tab:green'}
markers = {'KMeans': 's', 'DBSCAN': 'o', 'Birch': '^'}
xlabels = [f"1:{r}" for r in imbalance_ratios]
metrics = ['ari', 'silhouette', 'n_clusters']

os.makedirs("part_two", exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(7, 5))
    for alg in results:
        vals = np.array(results[alg][metric], dtype=np.float64)  # shape: (len(imbalance_ratios), n_runs)
        means = np.nanmean(vals, axis=1)
        stds = np.nanstd(vals, axis=1)
        plt.errorbar(
            xlabels, means, yerr=stds,
            marker=markers[alg], color=colors[alg], label=alg,
            linewidth=2, markersize=8, capsize=5
        )
        plt.scatter(xlabels, means, color=colors[alg], marker=markers[alg], s=80)
    plt.xlabel('Cluster Size Ratio (smallest:largest)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} vs Cluster Size Imbalance (mean ± std)")
    plt.grid(True, linestyle='--', alpha=0.6)
    if metric == 'ari':
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"part_two/imbalance_{metric}_results.png", bbox_inches="tight", dpi=300)
    plt.close()