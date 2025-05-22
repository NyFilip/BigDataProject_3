import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import dataSet as DS

def K_Elbow(images, k_range=range(1, 10), random_state=0, title='Elbow Method', plot=True, return_distortion=False):
    """
    Compute and optionally plot the inertia or distortion values for a range of k in KMeans.

    Parameters:
        images (array-like): The input data.
        k_range (range): Range of cluster counts to try.
        random_state (int): Random seed for reproducibility.
        title (str): Title for the plot.
        plot (bool): Whether to show the plot.
        return_distortion (bool): If True, return average distortion (mean distance to closest cluster center) instead of inertia.

    Returns:
        list: Inertia or distortion values for each k.
    """
    values = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(images)
        if return_distortion:
            # Compute mean distance to closest cluster center (distortion)
            distances = np.min(model.transform(images), axis=1)
            distortion = np.mean(distances)
            values.append(distortion)
        else:
            # Inertia (sum of squared distances to closest cluster center)
            values.append(model.inertia_)

    if plot:
        plt.figure(figsize=(6, 4))
        ylabel = 'Distortion' if return_distortion else 'Inertia'
        plt.plot(k_range, values, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return values

def run_kmeans(X, n_clusters=3, random_state=0):
    """
    Perform KMeans clustering on the dataset X.

    Parameters:
        X (array-like): Data matrix (samples x features).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.

    Returns:
        labels (ndarray): Cluster labels for each sample.
        model (KMeans): Fitted KMeans model.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return labels



def select_by_variance(X, threshold=0.1):
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X)
    return X_var, selector.get_support(indices=True)

def select_by_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def Select_by_tsne(
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
        max_iter=n_iter,
        random_state=random_state,
        method=method,
        **kwargs
    )
    return tsne.fit_transform(X)


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

def catdog_Kmeans():


    _, catdog_labels, sImagesMatrix, sImagesList = DS.catdog()
    X_catdog = sImagesList  # shape: (198, 4096

    

    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_catdog)
    kmeans = KMeans(n_clusters=2, random_state=0)

    labels = kmeans.fit_predict(X_pca)
    distance =kmeans.fit_transform(X_pca)
    return labels,distance

def mnist_Kmeans():

    _, mnist_labels, sImagesMatrix, sImagesList = DS.mnist()
    X_catdog = sImagesList  # shape: (198, 4096

    

    pca = PCA(n_components=26)
    X_pca = pca.fit_transform(X_catdog)
    kmeans = KMeans(n_clusters=9, random_state=0)

    labels = kmeans.fit_predict(X_pca)
    return labels

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_clusters(X, labels, pca_components=50, perplexity=30, learning_rate=200, random_state=42):
    """
    Scales data, reduces with PCA, applies t-SNE, and visualizes 2D clusters.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), e.g. flattened images
    - labels: ndarray of shape (n_samples,), e.g. cluster labels
    - pca_components: number of PCA components (default: 50)
    - perplexity: t-SNE perplexity (default: 30)
    - learning_rate: t-SNE learning rate (default: 200)
    - random_state: for reproducibility (default: 42)
    """
    print("[1/4] Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[2/4] Reducing dimensions with PCA...")
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    print("[3/4] Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                learning_rate=learning_rate, random_state=random_state)
    X_tsne = tsne.fit_transform(X_pca)

    print("[4/4] Plotting...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=40, alpha=0.7)
    plt.title("t-SNE Visualization of Clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




