from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np


def birch(dataset, clusters):
    #clusters = len(set(labels))
    
    brc = Birch(n_clusters = clusters)
    brc.fit(dataset)
    prediction = brc.predict(dataset)
    distance = brc.transform(dataset)
    print(f'predictions:{prediction.shape}, distance:{distance.shape}')
    plt.figure()
    plt.scatter(distance[0],distance[1], marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Birch Clustering')
    
    return prediction, distance


