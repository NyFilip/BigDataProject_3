import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster

from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
def K_Elbow(images, k_range=range(1,20),random_state=0):


    inertias = []
    
    for k in k_range:
        model = sklearn.cluster.KMeans(n_clusters=k)
        model.fit(images)
        inertias.append(model.inertia_)

    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()


def select_by_variance(X, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X)
    return X_var, selector.get_support(indices=True)


def select_by_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def FTestFeatureSelection(data, n_features=100,return_indices=False):
    """
    Perform F-test feature selection and return filtered dataset with labels.
    
    Parameters:
        data (numpy.ndarray): Dataset with labels in the first column.
        n_features (int): Number of top features to select based on F-statistic.
    
    Returns:
        filtered_data (numpy.ndarray): Dataset with labels and selected features.
    """
    images = data[:, 1:]  # Features
    labels = data[:, 0]   # Labels

    f_values, _ = f_classif(images, labels)
    # Get indices of the top n features based on F-statistic
    top_features = np.argsort(f_values)[-n_features:]  # Select top n features
    top_features = np.sort(top_features)  # Sort indices to maintain column order

    # Filter the features and add labels back as the first column
    filtered_features = images[:, top_features]
    filtered_data = np.column_stack((labels, filtered_features))
    if return_indices:
        return filtered_data, top_features
    return filtered_data


def select_by_TSNE(
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

