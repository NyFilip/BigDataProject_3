import Fkod as F
import Hkod as H
import Nkod as N
import Rkod as R
import dataSet as DS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# Load CatDog data
catdog_full, catdog_labels, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList  # shape: (198, 4096
mnist_full, mnist_labels, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()  # shape: (N, 256)






def Find_number_of_classes(X_catdog,X_mnist):
# find the best number of clusters for the datasets
#Kvalues to examine:
    k_range_Elbow=range(1,15)
    k_range_Silhoutte=range(2,15)
    #Elbow score for catdog:
    CatdoG_Elbow = F.K_Elbow(X_catdog, k_range_Elbow, random_state=42, plot=True,title='Elbow method for CatDog',return_distortion=False)
    #Elbow score for mnist:
    Mnist_Elbow = F.K_Elbow(X_mnist, k_range_Elbow, random_state=42, plot=True,title='Elbow method for MNIST',return_distortion=False)
    #silhouette score for catdog:
    CatDog_Silhoutte = H.silhouette_kmeans_analysis(X_catdog, k_range_Silhoutte, title='Silhouette Score for CatDog', random_state=42, plot=True)
    #silhouette score for mnist:
    MNIST_Silhoutte = H.silhouette_kmeans_analysis(X_mnist, k_range_Silhoutte, title='Silhouette Score for MNIST', random_state=42, plot=True)
    print("CatDog Silhouette Scores: ", CatDog_Silhoutte)
    print("MNIST Silhouette Scores: ", MNIST_Silhoutte) 
    print("Elbow Score for CatDog: ", CatdoG_Elbow)
    print("Elbow Score for MNIST: ", Mnist_Elbow)

if __name__ == '__main__':
    Find_number_of_classes(X_catdog,X_mnist)

    #getting labels from classifiers.

    # catdog

    catdog_birch_labels, catdog_birch_distance, = N.catdogBirch()
    catdog_Kmeans_labels, catdog_Kmeans_distance, = F.catdog_Kmeans()
    catdog_DBSCAN_labels, catdog_DBSCAN_distance, = R.catdogDbscan()
    
    mnist_birch_labels, mnist_birch_distance = N.mnistBirch()
    mnist_Kmeans_labels, mnist_Kmeans_distance = F.mnist_Kmeans()
    mnist_DBSCAN_labels, mnist_DBSCAN_distance = R.mnistDbscan()
    
    N.truePredPlot(catdog_full[:,1:], catdog_labels, catdog_birch_distance, catdog_birch_labels)
    N.truePredPlot(catdog_full[:,1:], catdog_labels, catdog_Kmeans_distance, catdog_Kmeans_labels)
    N.truePredPlot(catdog_full[:,1:], catdog_labels, catdog_DBSCAN_distance, catdog_DBSCAN_labels)
    
    N.truePredPlot(mnist_full[:,1:], mnist_labels, mnist_birch_distance, mnist_birch_labels)
    N.truePredPlot(mnist_full[:,1:], mnist_labels, mnist_Kmeans_distance, mnist_Kmeans_labels)
    N.truePredPlot(mnist_full[:,1:], mnist_labels, mnist_DBSCAN_distance, mnist_DBSCAN_labels)
    
    
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
    
            
