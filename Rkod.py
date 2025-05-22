import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from itertools import product

def plot_k_distance(data, k=5):
    # Plot k-distance (distance to k-th nearest neighbor) to help choose DBSCAN eps.
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
    
    # Resampling stability test for any clustering function.
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

def generate_mnist_like_data(
    n_clusters=10,
    cluster_sizes=None,  # list of length n_clusters
    n_features=50,       # higher dimension like MNIST (28x28=784)
    cluster_std=1.0,
    random_state=42
):
    if cluster_sizes is None:
        total_samples = 10000
        samples_per_cluster = total_samples // n_clusters
        cluster_sizes = [samples_per_cluster] * n_clusters
    else:
        total_samples = sum(cluster_sizes)

    X, y = [], []

    # Spread cluster centers far apart for well-separated clusters
    centers = np.random.RandomState(random_state).uniform(-20, 20, size=(n_clusters, n_features))

    for i in range(n_clusters):
        x_i, _ = make_blobs(
            n_samples=cluster_sizes[i],
            centers=[centers[i]],
            cluster_std=cluster_std,
            n_features=n_features,
            random_state=random_state + i
        )
        X.append(x_i)
        y.append(np.full(cluster_sizes[i], i))

    X = np.vstack(X)
    y = np.concatenate(y)
    X = StandardScaler().fit_transform(X)  # normalize

    return X, y

def evaluate_clustering(data, labels_pred, labels_true=None):
    if len(set(labels_pred)) < 2 or all(label == -1 for label in labels_pred):
        return -1  # invalid cluster scenario
    if labels_true is not None:
        return adjusted_rand_score(labels_true, labels_pred)
    else:
        return silhouette_score(data, labels_pred)

def auto_tune_dbscan(data, pca_range, eps_range, min_samples_range, labels_true=None):
    best_score = -1
    best_params = {}
    
    for n_components, eps, min_samples in product(pca_range, eps_range, min_samples_range):
        data_pca, _ = perform_pca(data, n_components=n_components)
        labels_pred = perform_dbscan(data_pca, eps=eps, min_samples=min_samples)
        
        score = evaluate_clustering(data_pca, labels_pred, labels_true)
        
        print(f"PCA: {n_components}, eps: {eps}, min_samples: {min_samples}, Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_params = {
                'n_components': n_components,
                'eps': eps,
                'min_samples': min_samples,
                'score': score
            }

    return best_params