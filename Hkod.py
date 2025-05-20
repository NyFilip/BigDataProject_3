from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
def silhouette_kmeans_analysis(
    X, 
    k_values, 
    title='Silhouette Score vs. Number of Clusters (k)', 
    random_state=42,
    plot=True
):
    """
    Computes silhouette scores for KMeans clustering over multiple k values.

    Parameters:
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        k_values : list of int
            Values of k (number of clusters) to evaluate
        title : str
            Title for the plot
        random_state : int
            Random seed for KMeans reproducibility
        plot : bool
            Whether to display the plot

    Returns:
        sil_scores : list of float
            Silhouette scores corresponding to each k in k_values
    """
    sil_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_values, sil_scores, marker='o')
        plt.title(title)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()

    return sil_scores