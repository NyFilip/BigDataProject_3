import numpy as np
import matplotlib.pyplot as plt
from Rkod import perform_dbscan, perform_pca, plot_clusters, plot_k_distance, test_stability, auto_tune_dbscan, evaluate_clustering
from dataSet import mnist, catdog

def run_experiment(data, labels_true, name, eps_values, min_samples_list, pca_components_list):
    print(f"\n--- Running experiment on {name} ---")

    # Step 1: Tune parameters
    best_params = auto_tune_dbscan(
        data,
        pca_range=pca_components_list,
        eps_range=eps_values,
        min_samples_range=min_samples_list,
        labels_true=labels_true
    )

    print(f"Best parameters for {name}: {best_params}")

    # Step 2: Perform DBSCAN with best params
    data_pca, _ = perform_pca(data, n_components=best_params["n_components"])
    labels_pred = perform_dbscan(data_pca, eps=best_params["eps"], min_samples=best_params["min_samples"])

    # Step 3: Plot result
    plot_clusters(data_pca, labels_pred, title=f"{name} - DBSCAN Clusters")

    # Step 4: Evaluate ARI
    score = evaluate_clustering(data_pca, labels_pred, labels_true)
    print(f"Final ARI for {name}: {score:.3f}")

    # Step 5: Stability test
    def cluster_fn_for_stability(data_subset):
        reduced, _ = perform_pca(data_subset, n_components=best_params["n_components"])
        return perform_dbscan(reduced, eps=best_params["eps"], min_samples=best_params["min_samples"])

    test_stability(
        data,
        cluster_fn=cluster_fn_for_stability,
        labels_ref=labels_pred,
        title=f"{name} - Clustering Stability"
    )

if __name__ == "__main__":
    # Load MNIST-like dataset
    from dataSet import mnist, catdog

    mnist_full, mnist_labels, mnist_matrix, mnist_list = mnist()
    print(f"MNIST-like data shape: {mnist_matrix.shape}")

    # Load Cat-Dog dataset
    catdog_full, catdog_labels, catdog_matrix, catdog_list = catdog()
    print(f"Cat-Dog data shape: {catdog_list.shape}")

    # Parameter ranges
    eps_values = np.arange(2.5, 3.6, 0.1)   # finer search between 2.5 and 3.5
    min_samples_list = [2, 3, 4, 5, 6, 7, 8]
    pca_components_list = [20, 25, 30, 35, 40]  # slightly extended around 30

    # Run experiments
    run_experiment(mnist_list, mnist_labels, "MNIST-like", 1.5, 10, 20)
    run_experiment(catdog_list, catdog_labels, "Cat-Dog", 1.5, 10, 20)
