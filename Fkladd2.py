import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dataSet as DS

# Suppress BIRCH subcluster warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Utility: Hungarian mapping for clusters -> labels
def best_cluster_mapping(true_labels, cluster_labels):
    labels = np.unique(true_labels)
    clusters = np.unique(cluster_labels)
    cost_matrix = np.zeros((len(labels), len(clusters)), dtype=int)
    for i, lab in enumerate(labels):
        for j, clu in enumerate(clusters):
            cost_matrix[i, j] = np.sum((true_labels == lab) & (cluster_labels == clu))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return {clusters[c]: labels[r] for r, c in zip(row_ind, col_ind)}

# Metrics: accuracy (with mapping) and ARI
def clustering_accuracy(true, pred):
    mapping = best_cluster_mapping(true, pred)
    mapped = np.array([mapping.get(c, -1) for c in pred])
    return accuracy_score(true, mapped)

# Unsupervised feature selection grid (no labels used)
def feature_selection_methods(X):
    methods = {}
    for n_comp in [5, 10, 20, 50, 100]:
        methods[f'pca_{n_comp}'] = PCA(n_components=n_comp)
        methods[f'kpca_rbf_{n_comp}'] = KernelPCA(n_components=n_comp, kernel='rbf')
    return methods

# Clustering methods grid
def clustering_methods(k):
    methods = {}
    for n_init in [10, 20]:
        methods[f'kmeans_{k}_{n_init}'] = KMeans(n_clusters=k, n_init=n_init)
    for eps in [0.5, 1.0, 1.5]:
        for min_samples in [3, 5, 10]:
            methods[f'dbscan_{eps}_{min_samples}'] = DBSCAN(eps=eps, min_samples=min_samples)
    for threshold in [0.3, 0.5, 0.7]:
        for bf in [20, 50]:
            methods[f'birch_{threshold}_{bf}'] = Birch(n_clusters=k, threshold=threshold, branching_factor=bf)
    return methods

# Grid search without supervised FS

def run_grid_search(X, y, n_clusters):
    feats = feature_selection_methods(X)
    clusts = clustering_methods(n_clusters)
    results = []
    for fname, fs in feats.items():
        X_fs = fs.fit_transform(X)
        for cname, clf in clusts.items():
            pred = clf.fit_predict(X_fs)
            acc = clustering_accuracy(y, pred)
            ari = adjusted_rand_score(y, pred)
            results.append({'feat': fname, 'cluster': cname, 'accuracy': acc,
                            'ari': ari, 'X_fs': X_fs, 'clf': clf})
    return sorted(results, key=lambda x: x['accuracy'], reverse=True)

# Visualization utility
def visualize_embedding(X, labels, title):
    emb = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(6,6))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(emb[idx,0], emb[idx,1], label=str(lab), alpha=0.6)
    plt.legend(title='Cluster')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()

# Evaluate & plot best of each algorithm

def evaluate_top_algorithms(results, dataset_name):
    for alg in ['kmeans', 'dbscan', 'birch']:
        best = max([r for r in results if r['cluster'].startswith(alg)],
                   key=lambda x: x['accuracy'], default=None)
        if best:
            print(f"{dataset_name} - Best {alg.upper()}")
            print(f"  Feature: {best['feat']}")
            print(f"  Params: {best['cluster']}")
            print(f"  Accuracy: {best['accuracy']:.3f}")
            print(f"  ARI: {best['ari']:.3f}\n")
            viz_title = f"{dataset_name} - {alg.upper()} ({best['feat']})"
            visualize_embedding(best['X_fs'], best['clf'].fit_predict(best['X_fs']), viz_title)

# Load & flatten
_, labels_cat, X_cat_raw, _ = DS.catdog()
_, labels_mnist, X_mnist_raw, _ = DS.mnist()
X_cat = X_cat_raw.reshape(X_cat_raw.shape[0], -1)
X_digits = X_mnist_raw.reshape(X_mnist_raw.shape[0], -1)

# Run for MNIST
print("\n=== MNIST Results (Unsupervised FS) ===")
results_digits = run_grid_search(X_digits, labels_mnist, n_clusters=9)
evaluate_top_algorithms(results_digits, 'Digits')

# Run for Cats vs Dogs
print("\n=== Cats vs Dogs Results (Unsupervised FS) ===")
results_pets = run_grid_search(X_cat, labels_cat, n_clusters=2)
evaluate_top_algorithms(results_pets, 'Pets')
