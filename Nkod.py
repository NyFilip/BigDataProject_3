from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np


def birch(dataset, labels, thresh = 0.5):
    clusters = len(set(labels))
    
    brc = Birch(threshold = thresh, n_clusters = clusters)
    brc.fit(dataset)
    prediction = brc.predict(dataset)
    distance = brc.fit_transform(dataset)
    #print(clusters)
    #print(f'predictions:{prediction.shape}, distance:{distance.shape}')
    #print(dataset.shape)
    setLength = len(distance[1])
    
    plt.figure()
    plt.scatter(dataset[:setLength,0], distance[:setLength,1], c=prediction[:setLength], marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Birch Clustering')
    
    return prediction, distance


