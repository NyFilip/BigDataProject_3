from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np


def birch(dataset, labels):
    clusters = len(set(labels))
    
    brc = Birch(n_clusters = clusters)
    brc.fit(dataset)
    prediction = brc.predict(dataset)
    distance = brc.fit_transform(dataset)
    print(clusters)
    print(f'predictions:{prediction.shape}, distance:{distance.shape}')
    setLength = len(distance[1])
    plt.figure()
    plt.scatter(distance[0], distance[1], c=prediction[:setLength], marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Birch Clustering')
    
    return prediction, distance


