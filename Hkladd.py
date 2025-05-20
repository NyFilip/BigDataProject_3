from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os
import numpy as np
import matplotlib.pyplot as plt

def mnist(filePath = 'Numbers.txt'):
    imagesList = []
    imagesMatrix = []
    labels = []
    full = []
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            index = parts[0].strip()
            label = np.array(int(parts[1]))
            fullTemp  = np.array(list(map(float,parts[1:])))
            pixels = list(map(float,parts[2:]))
            
            pixelArray = np.array(pixels).reshape(16,16)
            normArray = (pixelArray + 1) / 2
              
            
            labels.append(label)
            imagesMatrix.append(normArray)
            full.append(fullTemp)
            imagesList.append(np.array(normArray).reshape(1,256))
    
    full = np.array(full)
    labels = np.array(labels)
    imagesMatrix = np.array(imagesMatrix)
    imagesList = np.array(imagesList)                    
    return full, labels, imagesMatrix, imagesList
            
def catdog(filePath = 'catdogdata.txt'):
    labels = []
    labels[:98] = np.zeros(99)
    labels[99:] = np.ones(99)
    sLabels = []
    sImagesMatrix = []
    sImagesList = []
    full = []
    i = 0
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            index = parts[0].strip()
            pixels = list(map(int,parts[1:]))

            fullTemp = list(pixels)
            if i < 99:
                fullTemp.insert(0, 0)
            else:
                fullTemp.insert(0,1)
            full.append(fullTemp)
            i += 1
    full = np.array(full)
    np.random.seed(10)
    np.random.shuffle(full)

    for line in full:
        slabel = line[0]
        spixels = line[1:]
        sLabels.append(slabel)
        sImagesList.append(spixels)
        sImagesMatrix.append(np.array(spixels).reshape(64,64))
    sLabels = np.array(sLabels)
    sImagesList = np.array(sImagesList)
    sImagesMatrix = np.array(sImagesMatrix)
    
    
    return full, sLabels, sImagesMatrix, sImagesList

def visCatDog(cdMatrix):
    fig, axs = plt.subplots(4,4)
    for i in range(16):
        axs[int(np.floor((i)/4)),(i)%4].matshow(cdMatrix[np.random.randint(0, high=198)])
    plt.show()

def visMnist(mMatrix):
    fig, axs = plt.subplots(4,4)
    for i in range(16):
        axs[int(np.floor((i)/4)),(i)%4].matshow(mMatrix[np.random.randint(0, high=2000)])
    plt.show()

'''# Load MNIST data
_, _, imagesMatrix, imagesList = mnist()

X_mnist = imagesList.squeeze()  # shape: (N, 256)

k_values = range(2, 11)  # Try k = 2 to 10
sil_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_mnist)
    score = silhouette_score(X_mnist, cluster_labels)
    sil_scores.append(score)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(k_values, sil_scores, marker='o')
plt.title('MNIST: Silhouette Score vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

'''

# Load CatDog data
_, _, sImagesMatrix, sImagesList = catdog()

X_catdog = sImagesList  # shape: (198, 4096)

k_values = range(2, 8)  # Try small values of k, since we expect 2 clusters
sil_scores_cd = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_catdog)
    score = silhouette_score(X_catdog, cluster_labels)
    sil_scores_cd.append(score)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(k_values, sil_scores_cd, marker='s')
plt.title('CatDog: Silhouette Score vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
