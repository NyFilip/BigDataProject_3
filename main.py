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

# find the best number of clusters for the datasets
#Kvalues to examine:
k_range_Elbow=range(1,20)
k_range_Silhoutte=range(2,20)
#Elbow score for catdog:
CatdoG_Elbow = H.K_Elbow(X_catdog, k_range_Elbow, random_state=42, plot=True)
#Elbow score for mnist:
Mnist_Elbow = H.K_Elbow(X_mnist, k_range_Elbow, random_state=42, plot=True)
#silhouette score for catdog:
CatDog_Silhoutte = H.silhouette_kmeans_analysis(X_catdog, k_range_Silhoutte, title='Silhouette Score for CatDog', random_state=42, plot=True)
#silhouette score for mnist:
MNIST_Silhoutte = H.silhouette_kmeans_analysis(X_mnist, k_range_Silhoutte, title='Silhouette Score for MNIST', random_state=42, plot=True)
print("CatDog Silhouette Scores: ", CatDog_Silhoutte)
print("MNIST Silhouette Scores: ", MNIST_Silhoutte) 
print("Elbow Score for CatDog: ", CatdoG_Elbow)
print("Elbow Score for MNIST: ", Mnist_Elbow)
