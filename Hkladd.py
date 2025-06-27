import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import dataSet as DS
import Nkod as N
import Fkod as F
import Rkod as R    

# Load datasets
_, catdog_labels, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList
_, mnist_labels, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()

# === Clustering stability functions === #
def clustering_stability(X, clustering_func, n_trials=20, sample_frac=0.8, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    all_labels = []

    for _ in range(n_trials):
        indices = np.random.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        subset = X[indices]
        labels = clustering_func(subset)
        all_labels.append((indices, labels))

    scores = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            idx_i, labels_i = all_labels[i]
            idx_j, labels_j = all_labels[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 5:
                continue
            map_i = {idx: label for idx, label in zip(idx_i, labels_i)}
            map_j = {idx: label for idx, label in zip(idx_j, labels_j)}
            common_labels_i = [map_i[idx] for idx in common]
            common_labels_j = [map_j[idx] for idx in common]
            scores.append(adjusted_rand_score(common_labels_i, common_labels_j))

    return scores

# === Define clustering function wrappers === #
def kmeans_func(X):
    return F.run_kmeans(X, k=9, random_state=42)

def birch_func(X):
    return N.birch(X, k=9, threshold=1.7)[0]

def dbscan_func(X):
    return R.perform_dbscan(X, eps=3.2, min_samples=2)[0]

# === Run stability analysis with 5 seeds === #
seeds = [10, 20, 30, 40, 50]
all_results = {"KMeans": [], "Birch": [], "DBSCAN": []}

for seed in seeds:
    scores_k = clustering_stability(X_mnist, lambda X: F.run_kmeans(X, k=9, random_state=seed), random_state=seed)
    scores_b = clustering_stability(X_mnist, lambda X: N.birch(X, k=9, thresh=1.7)[0], random_state=seed)
    scores_d = clustering_stability(X_mnist, lambda X: R.perform_dbscan(X, eps=3.2, min_samples=2)[0], random_state=seed)

    all_results["KMeans"].extend(scores_k)
    all_results["Birch"].extend(scores_b)
    all_results["DBSCAN"].extend(scores_d)

# === Plotting boxplot of stability scores === #
plt.figure(figsize=(10, 6))
plt.boxplot([all_results["KMeans"], all_results["Birch"], all_results["DBSCAN"]],
            labels=["KMeans", "Birch", "DBSCAN"])
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("Clustering Stability (Variability over 5 Runs × 20 Trials)")
plt.grid(True)
plt.tight_layout()
plt.show()
