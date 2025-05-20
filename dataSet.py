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