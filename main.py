import Fkod as F
import Hkod as H
import Nkod as N
import Rkod as R
import dataSet as DS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# Load CatDog data
_, catdog_labels, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList  # shape: (198, 4096
_, mnist_labels, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()  # shape: (N, 256)






def Find_number_of_classes(X_catdog,X_mnist):
# find the best number of clusters for the datasets
#Kvalues to examine:
    k_range_Elbow=range(1,15)
    k_range_Silhoutte=range(2,15)
    #Elbow score for catdog:
    CatdoG_Elbow = F.K_Elbow(X_catdog, k_range_Elbow, random_state=42, plot=True,title='Elbow method for CatDog',return_distortion=False)
    #Elbow score for mnist:
    Mnist_Elbow = F.K_Elbow(X_mnist, k_range_Elbow, random_state=42, plot=True,title='Elbow method for MNIST',return_distortion=True)
    #silhouette score for catdog:
    CatDog_Silhoutte = H.silhouette_kmeans_analysis(X_catdog, k_range_Silhoutte, title='Silhouette Score for CatDog', random_state=42, plot=True)
    #silhouette score for mnist:
    MNIST_Silhoutte = H.silhouette_kmeans_analysis(X_mnist, k_range_Silhoutte, title='Silhouette Score for MNIST', random_state=42, plot=True)
    print("CatDog Silhouette Scores: ", CatDog_Silhoutte)
    print("MNIST Silhouette Scores: ", MNIST_Silhoutte) 
    print("Elbow Score for CatDog: ", CatdoG_Elbow)
    print("Elbow Score for MNIST: ", Mnist_Elbow)


# Find_number_of_classes(X_catdog_PCA,X_mnist)

#getting labels from classifiers.

# catdog

catdog_birch_labels,catdog_birch_distance=N.catdogBirch()
catdog_Kmeans_labels,catdog_Kmeans_distance =F.catdog_Kmeans()
# catdog_DBSCAN_labels,catdog_DBSCAN_disctance =R.catdogDbscan()

N.truePredPlot(X_catdog,catdog_labels,catdog_birch_distance,catdog_birch_labels)
N.truePredPlot(X_catdog,catdog_labels,catdog_Kmeans_distance,catdog_Kmeans_labels)
plt.show()

# F.evaluate_clustering(catdog_labels,catdog_birch_labels)
# F.evaluate_clustering(catdog_labels,catdog_Kmeans_labels)
# F.evaluate_clustering(catdog_labels,catdog_DBSCAN_labels)

# # mnist

# mnist_birch_labels,_=N.mnistBirch()
# mnist_Kmeans_labels =F.mnist_Kmeans()
# mnist_DBSCAN_labels =R.mnistDbscan()

# F.evaluate_clustering(mnist_labels,mnist_birch_labels)
# F.evaluate_clustering(mnist_labels,mnist_Kmeans_labels)
# F.evaluate_clustering(mnist_labels,mnist_DBSCAN_labels)
# F.evaluate_clustering(y_true=mnist_labels,y_pred=kMeans_mnist_labels)


