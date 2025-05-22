import Fkod as F
import Hkod as H
import Nkod as N
import Rkod as R
import dataSet as DS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Load CatDog data
_, _, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList  # shape: (198, 4096
_, _, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()  # shape: (N, 256)

# Filer data 

X_mnist_PCA,_=F.select_by_pca(X_mnist,n_components=30)
X_catdog_PCA,_=F.select_by_pca(X_catdog,n_components=30)

X_mnist_TSNE=F.Select_by_tsne(X_mnist_PCA,perplexity=30,learning_rate=200,n_iter=1000)
X_catdog_TSNE=F.Select_by_tsne(X_catdog_PCA,perplexity=30,learning_rate=200,n_iter=1000)

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

## Cluster the data

kMeans_mnist_labels= F.run_kmeans(X_mnist_TSNE,clusters=9)
kMeans_catdog_labels=F.run_kmeans()
birch_labels=N.birch()

