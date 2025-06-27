import numpy as np
import matplotlib.pyplot as plt
import dataSet as DS
import Nkod as N
import Fkod as F
import Rkod as R

# === Custom ARI implementation === #
def adjusted_rand_index(labels_true, labels_pred):
    from collections import Counter
    from math import comb

    contingency = {}
    for t, p in zip(labels_true, labels_pred):
        contingency.setdefault((t, p), 0)
        contingency[(t, p)] += 1

    sum_comb_cij = sum(comb(n_ij, 2) for n_ij in contingency.values())
    a = Counter(labels_true)
    b = Counter(labels_pred)
    sum_comb_ai = sum(comb(n_i, 2) for n_i in a.values())
    sum_comb_bj = sum(comb(n_j, 2) for n_j in b.values())

    n = len(labels_true)
    if n < 2:
        return 0.0

    total_pairs = comb(n, 2)
    expected_index = (sum_comb_ai * sum_comb_bj) / total_pairs
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    index = sum_comb_cij

    if max_index == expected_index:
        return 0.0
    return (index - expected_index) / (max_index - expected_index)

# === Generic clustering stability function === #
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
            scores.append(adjusted_rand_index(common_labels_i, common_labels_j))

    return scores

# === Define clustering function factories (wrappers with seed) === #
def kmeans_func(seed):
    return lambda X: F.run_kmeans(X, k=9, random_state=seed)

def birch_func(seed):
    return lambda X: N.birch(X, k=9, thresh=1.7)[0]

def dbscan_func(seed):
    return lambda X: R.perform_dbscan(X, eps=3.2, min_samples=2)[0]

# === Load dataset (MNIST or CatDog) === #
_, _, _, X_mnist = DS.mnist()
X_mnist = X_mnist.squeeze()

# === Run clustering stability with multiple seeds === #
seeds = [10, 20, 30, 40, 50]
all_results = {"KMeans": [], "Birch": [], "DBSCAN": []}

for seed in seeds:
    print(f"Running seed {seed}...")
    scores_k = clustering_stability(X_mnist, kmeans_func(seed), random_state=seed)
    scores_b = clustering_stability(X_mnist, birch_func(seed), random_state=seed)
    scores_d = clustering_stability(X_mnist, dbscan_func(seed), random_state=seed)

    all_results["KMeans"].extend(scores_k)
    all_results["Birch"].extend(scores_b)
    all_results["DBSCAN"].extend(scores_d)

# === Plot boxplot of ARI scores === #
plt.figure(figsize=(10, 6))
plt.boxplot(
    [all_results["KMeans"], all_results["Birch"], all_results["DBSCAN"]],
    labels=["KMeans", "Birch", "DBSCAN"]
)
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("Clustering Stability (Manual ARI, 5 Seeds × 20 Trials)")
plt.grid(True)
plt.tight_layout()
plt.show()
