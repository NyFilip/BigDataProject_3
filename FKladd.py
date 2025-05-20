import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import dataSet

castanddogs= dataSet.catdog('catdogdata.txt')[0]
labels=castanddogs[0,:]
images=castanddogs[1:,:]

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


K_Elbow(images=images)