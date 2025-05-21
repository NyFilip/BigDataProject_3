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

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    cm  = confusion_matrix(y_true, y_pred)
    purity = np.sum(np.max(cm, axis=1)) / np.sum(cm)

    print(f"ARI  : {ari:.3f}")
    print(f"NMI  : {nmi:.3f}")
    print(f"Purity: {purity:.3f}")
    print("Confusion matrix:\n", cm)
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

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data where each row is a flattened image.
    n_components : int, default=2
        Target number of dimensions.
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    learning_rate : float, default=200.0
        The learning rate for t-SNE optimization.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization.
    random_state : int, default=42
        Seed for reproducibility.
    **kwargs : dict
        Any additional keyword arguments to pass to sklearn.manifold.TSNE.

    Returns
    -------
    embedding : np.ndarray, shape (n_samples, n_components)
        The t-SNE embedding of the input data.
    """
    # Validate input
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be 2D: shape (n_samples, n_features)")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        **kwargs
    )
    embedding = tsne.fit_transform(X)
    return embedding


castanddogs= dataSet.catdog('catdogdata.txt')[0]
castanddogs= dataSet.mnist('numbers.txt')[0]
labels=castanddogs[:,0]
images=castanddogs[:,1:]
print(images.shape)
X_var, selected_indices = Fkod.select_by_variance(images, threshold=0.000001)
X_ftest=Fkod.FTestFeatureSelection(X_var,100)
X_pca, pca_model = Fkod.select_by_pca(images, n_components=80)


Y = tsne_embed(images, n_components=2, perplexity=40, n_iter=500)

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


