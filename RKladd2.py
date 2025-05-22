# Rkladd.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, Birch, DBSCAN

sns.set(style="whitegrid")


import numpy as np
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

def estimate_eps(X, k=10):
    """Estimate a good eps value for DBSCAN using k-NN distances."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return np.mean(np.sort(distances[:, -1]))

def apply_clustering(X, y_true=None):
    results = {}

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    results['KMEANS'] = {
        'labels': y_kmeans,
        'ari': adjusted_rand_score(y_true, y_kmeans) if y_true is not None else None,
        'silhouette': silhouette_score(X, y_kmeans) if len(set(y_kmeans)) > 1 else float('nan')
    }

    # Birch
    birch = Birch(n_clusters=3)
    y_birch = birch.fit_predict(X)
    results['BIRCH'] = {
        'labels': y_birch,
        'ari': adjusted_rand_score(y_true, y_birch) if y_true is not None else None,
        'silhouette': silhouette_score(X, y_birch) if len(set(y_birch)) > 1 else float('nan')
    }

    # DBSCAN with auto-tuned eps
    eps = estimate_eps(X, k=10)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    y_dbscan = dbscan.fit_predict(X)
    if len(set(y_dbscan)) > 1 and -1 in y_dbscan and len(set(y_dbscan)) == 2:
        silhouette = float('nan')
    else:
        silhouette = silhouette_score(X, y_dbscan) if len(set(y_dbscan)) > 1 else float('nan')
    results['DBSCAN'] = {
        'labels': y_dbscan,
        'ari': adjusted_rand_score(y_true, y_dbscan) if y_true is not None else None,
        'silhouette': silhouette
    }

    return results




def generate_mnist_like_data(n_clusters=10, cluster_sizes=None, n_features=50, cluster_std=1.0, random_state=None):
    if cluster_sizes is None:
        cluster_sizes = [500] * n_clusters
    assert len(cluster_sizes) == n_clusters, "Mismatch in number of clusters and cluster_sizes"

    X_all = []
    y_all = []
    centers = np.random.RandomState(random_state).randn(n_clusters, n_features) * 5

    for i, size in enumerate(cluster_sizes):
        X, _ = make_blobs(n_samples=size, centers=[centers[i]], n_features=n_features,
                          cluster_std=cluster_std, random_state=random_state)
        X_all.append(X)
        y_all.extend([i] * size)

    return np.vstack(X_all), np.array(y_all)


def perform_pca(X, n_components=30):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def perform_clustering(method, X, n_clusters=10, eps=1.5, min_samples=5, random_state=42):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif method == 'birch':
        model = Birch(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError("Invalid method. Choose from 'kmeans', 'birch', 'dbscan'.")
    
    labels = model.fit_predict(X)
    return labels


def compare_methods_sample_size(
    test_sizes=[100, 200, 500, 1000, 2000, 3000],
    methods=['kmeans', 'birch', 'dbscan'],
    n_clusters=10,
    n_features=50,
    cluster_std=1.0,
    pca_components=30,
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    random_state=42
):
    results = []
    for size in test_sizes:
        cluster_sizes = [size // n_clusters] * n_clusters
        X, y = generate_mnist_like_data(
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=random_state
        )
        X_pca, _ = perform_pca(X, n_components=pca_components)

        for method in methods:
            labels_pred = perform_clustering(
                method,
                X_pca,
                n_clusters=n_clusters,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                random_state=random_state
            )
            ari = adjusted_rand_score(y, labels_pred)
            try:
                sil = silhouette_score(X_pca, labels_pred)
            except:
                sil = np.nan
            results.append({
                'Method': method.upper(),
                'Sample Size': size,
                'ARI': ari,
                'Silhouette': sil
            })
            print(f"Sample {size} | {method.upper()} | ARI: {ari:.3f} | Silhouette: {sil:.3f}")
    
    df = pd.DataFrame(results)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    sns.lineplot(data=df, x='Sample Size', y='ARI', hue='Method', marker='o', ax=axs[0])
    axs[0].set_title("ARI vs Sample Size")
    axs[0].grid(True)

    sns.lineplot(data=df, x='Sample Size', y='Silhouette', hue='Method', marker='o', ax=axs[1])
    axs[1].set_title("Silhouette vs Sample Size")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    return df


def compare_methods_imbalance(
    base_size=2000,
    imbalance_ratios=[1.0, 0.7, 0.5, 0.3, 0.1],
    methods=['kmeans', 'birch', 'dbscan'],
    n_clusters=10,
    n_features=50,
    cluster_std=1.0,
    pca_components=30,
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    random_state=42
):
    results = []
    for ratio in imbalance_ratios:
        major = base_size // 2
        cluster_sizes = [major if i < n_clusters - 1 else int(major * ratio) for i in range(n_clusters)]
        X, y = generate_mnist_like_data(
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=random_state
        )
        X_pca, _ = perform_pca(X, n_components=pca_components)

        for method in methods:
            labels_pred = perform_clustering(
                method,
                X_pca,
                n_clusters=n_clusters,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                random_state=random_state
            )
            ari = adjusted_rand_score(y, labels_pred)
            try:
                sil = silhouette_score(X_pca, labels_pred)
            except:
                sil = np.nan
            results.append({
                'Method': method.upper(),
                'Imbalance Ratio': ratio,
                'ARI': ari,
                'Silhouette': sil
            })
            print(f"Imbalance {ratio:.2f} | {method.upper()} | ARI: {ari:.3f} | Silhouette: {sil:.3f}")
    
    df = pd.DataFrame(results)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    sns.lineplot(data=df, x='Imbalance Ratio', y='ARI', hue='Method', marker='o', ax=axs[0])
    axs[0].set_title("ARI vs Imbalance")
    axs[0].grid(True)

    sns.lineplot(data=df, x='Imbalance Ratio', y='Silhouette', hue='Method', marker='o', ax=axs[1])
    axs[1].set_title("Silhouette vs Imbalance")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    return df


def run_all_comparisons():
    print("🔁 Running Sample Size Comparison...")
    df_size = compare_methods_sample_size()

    print("\n🔁 Running Imbalance Comparison...")
    df_imbalance = compare_methods_imbalance()

    print("\n📊 Sample Size Results:\n", df_size.pivot(index='Sample Size', columns='Method')[['ARI', 'Silhouette']])
    print("\n📊 Imbalance Results:\n", df_imbalance.pivot(index='Imbalance Ratio', columns='Method')[['ARI', 'Silhouette']])


if __name__ == "__main__":
    run_all_comparisons()