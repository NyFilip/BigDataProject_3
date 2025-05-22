import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Rkod import perform_dbscan, perform_pca
from dataSet import mnist, catdog

def run_simple_experiment(data, labels_true, name, eps, min_samples, n_components):
    print(f"\n--- Running simple experiment on {name} ---")

    # Step 1: Reduce dimensions with PCA
    data_pca, _ = perform_pca(data, n_components=n_components)

    # Step 2: Run DBSCAN
    labels_pred = perform_dbscan(data_pca, eps=eps, min_samples=min_samples)

    # Step 3: Display confusion matrix
    print(f"Predicted labels (unique): {np.unique(labels_pred)}")
    print(f"True labels (unique): {np.unique(labels_true)}")

    # Filter out noise (-1) for confusion matrix
    valid_idx = labels_pred != -1
    cm = confusion_matrix(labels_true[valid_idx], labels_pred[valid_idx])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title(f"{name} - Confusion Matrix (without noise)")
    plt.show()

if __name__ == "__main__":
    # Load datasets
    mnist_full, mnist_labels, mnist_matrix, mnist_list = mnist()
    catdog_full, catdog_labels, catdog_matrix, catdog_list = catdog()

    # Set fixed DBSCAN and PCA parameters
    eps = 3.0
    min_samples = 5
    n_components = 30

    # Run simplified experiments
    run_simple_experiment(mnist_list, mnist_labels, "MNIST-like", eps, min_samples, n_components)
    run_simple_experiment(catdog_list, catdog_labels, "Cat-Dog", eps, min_samples, n_components)
