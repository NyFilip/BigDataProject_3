import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import dataSet
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import Fkod
def evaluate_clustering(y_true, y_pred):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    cm  = confusion_matrix(y_true, y_pred)
    purity = np.sum(np.max(cm, axis=1)) / np.sum(cm)

    print(f"ARI  : {ari:.3f}")
    print(f"NMI  : {nmi:.3f}")
    print(f"Purity: {purity:.3f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
import numpy as np
from sklearn.manifold import TSNE

def tsne_embed(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensionality of flattened images via t-SNE.

    Automatically uses 'exact' method if n_components > 3.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be 2D: shape (n_samples, n_features)")

    method = kwargs.pop("method", "barnes_hut" if n_components <= 3 else "exact")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        method=method,
        **kwargs
    )
    return tsne.fit_transform(X)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

def tune_pca_kmeans(X, y_true, n_clusters, pca_range):
    """
    Finds the best number of PCA components for KMeans clustering by maximizing ARI.

    Parameters:
        X (ndarray): Data matrix (samples x features)
        y_true (ndarray): True labels
        n_clusters (int): Number of clusters for KMeans
        pca_range (iterable): Range of PCA components to try

    Returns:
        best_n (int): Best number of PCA components
        best_ari (float): Best ARI score
        best_labels (ndarray): Cluster labels for best run
    """
    best_ari = -1
    best_n = None
    best_labels = None

    for n in pca_range:
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(X_pca)
        ari = adjusted_rand_score(y_true, labels)
        # print(f"PCA components: {n}, ARI: {ari:.3f}")
        if ari > best_ari:
            best_ari = ari
            best_n = n
            best_labels = labels

    # print(f"\nBest PCA components: {best_n}, Best ARI: {best_ari:.3f}")
    return best_n, best_ari, best_labels

# Example usage:
# best_n, best_ari, best_labels = tune_pca_kmeans(images, labels, n_clusters=9, pca_range=range(5, 81, 5))


castanddogs= dataSet.catdog('catdogdata.txt')[0]
mnist= dataSet.mnist('numbers.txt')[0]
labels=castanddogs[:,0]
images=castanddogs[:,1:]


best_n, best_ari, best_labels = tune_pca_kmeans(images, labels, n_clusters=2, pca_range=range(1, 200, 1))
print(images.shape)
X_var, selected_indices = Fkod.select_by_variance(images, threshold=0.000001)
X_ftest=Fkod.FTestFeatureSelection(X_var,100)
X_pca, pca_model = Fkod.select_by_pca(images, n_components=80)


Y = tsne_embed(images, n_components=10, perplexity=40, n_iter=500)

# Quick scatter plot
plt.scatter(Y[:, 0], Y[:, 1])
plt.title("t-SNE Embedding of Flattened Images")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()










# Assume images is your data matrix (samples x features)
kmeans = KMeans(n_clusters=9, random_state=0)  # Set n_clusters as needed
kmeans_labels = kmeans.fit_predict(Y)

# Print cluster labels for each sample
print("KMeans cluster labels:", kmeans_labels)
print("trueLabels ", labels)

# Optional: Visualize clusters if data is 2D or reduced to 2D
# print(X_pca.shape[1])
# if X_pca.shape[1] == 2:
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
#     plt.title('KMeans Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.show()
# Fkod.K_Elbow(X_pca)

evaluate_clustering(labels,kmeans_labels)


