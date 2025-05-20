import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster

from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
def K_Elbow(images, k_range=range(1,10),random_state=0):


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
