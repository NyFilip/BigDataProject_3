import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from Rkod import generate_mnist_like_data, perform_dbscan
from Fkod import run_kmeans
from Nkod import birch

def evaluate_clustering(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

def run_experiment(sample_sizes, n_clusters=10, n_features=50, balanced=True):
    results = {
        'KMeans': [],
        'DBSCAN': [],
        'Birch': []
    }

    for size in sample_sizes:
        print(f"\nRunning for sample size {size}...")

        # Choose cluster size distribution
        if balanced:
            cluster_sizes = [size // n_clusters] * n_clusters
        else:
            rng = np.random.default_rng(42)
            min_per_cluster = 5
            if size < min_per_cluster * n_clusters:
                raise ValueError(f"Sample size {size} too small for {n_clusters} clusters with minimum {min_per_cluster} samples each.")

            proportions = rng.dirichlet(np.ones(n_clusters))
            cluster_sizes = (proportions * (size - min_per_cluster * n_clusters)).astype(int) + min_per_cluster
            cluster_sizes[-1] = size - np.sum(cluster_sizes[:-1])  # fix rounding



        # Generate synthetic data
        X, y_true = generate_mnist_like_data(
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            n_features=n_features,
            cluster_std=1.0,
            random_state=42
        )

        # Run KMeans
        y_kmeans = run_kmeans(X, n_clusters=n_clusters, random_state=42)
        results['KMeans'].append(evaluate_clustering(y_true, y_kmeans))

        # Run DBSCAN
        y_dbscan = perform_dbscan(X, eps=0.5, min_samples=5)
        results['DBSCAN'].append(evaluate_clustering(y_true, y_dbscan))

        # Run Birch
        y_birch, _ = birch(X, 9, thresh=0.5)
        results['Birch'].append(evaluate_clustering(y_true, y_birch))

    return results

def plot_results(sample_sizes, results):
    plt.figure(figsize=(10, 6))
    for method, scores in results.items():
        plt.plot(sample_sizes, scores, marker='o', label=method)
    plt.axhline(y=0.3, color='red', linestyle='--', label='Breakdown Threshold')
    plt.title('Clustering Performance vs. Sample Size')
    plt.xlabel('Total Sample Size')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run it for both balanced and unbalanced cases
sample_sizes = [10000, 5000, 2000, 1000, 500, 300, 200, 150, 100, 75, 50]

print("\n===== BALANCED CLUSTERS =====")
results_balanced = run_experiment(sample_sizes, balanced=True)
plot_results(sample_sizes, results_balanced)

print("\n===== UNBALANCED CLUSTERS =====")
results_unbalanced = run_experiment(sample_sizes, balanced=False)
plot_results(sample_sizes, results_unbalanced)
