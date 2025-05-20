import Fkod as F
import Hkod as H
import Nkod as N
import Rkod as R
import dataSet as DS

# Load CatDog data
_, _, sImagesMatrix, sImagesList = DS.catdog()
X_catdog = sImagesList  # shape: (198, 4096
_, _, imagesMatrix, imagesList = DS.mnist()
X_mnist = imagesList.squeeze()  # shape: (N, 256)
