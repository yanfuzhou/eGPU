# Reference: https://colab.research.google.com/github/FreeOfConfines/ExampleNNWithKerasAndTensorflow/blob/master/K_Nearest_Neighbor_Classification_with_Tensorflow_on_Fashion_MNIST_Dataset.ipynb#scrollTo=WEy4WAsBzFVo
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Download Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

print("--------------------------")
print("Dimensions of Train Set")
print("Dimension(trImages)=",np.shape(trImages))
print("There are", np.shape(trImages)[0], "images where each image is", np.shape(trImages)[1:], "in size")
print("There are", np.shape(np.unique(tLabels))[0], "unique image labels")
print("--------------------------")
print("Dimensions of Test Set")
print("Dimension(tImages)=",np.shape(tImages), "Dimension(tLabels)=", np.shape(tLabels)[0])
print("--------------------------")

paramk = 11  # parameter k of k-nearest neighbors
numTrainImages = np.shape(trLabels)[0]  # so many train images
numTestImages = np.shape(tLabels)[0]  # so many test images

arrayKNNLabels = np.array([])
numErrs = 0
for iTeI in range(0, numTestImages):
    arrayL2Norm = np.array([])  # store distance of a test image from all train images

    tmpTImage = np.copy(tImages[iTeI])
    tmpTImage[tmpTImage > 0] = 1

    for jTrI in range(numTrainImages):
        tmpTrImage = np.copy(trImages[jTrI])
        tmpTrImage[tmpTrImage > 0] = 1

        l2norm = np.sum(((tmpTrImage - tmpTImage) ** 2) ** (
            0.5))  # distance between two images; 255 is max. pixel value ==> normalization
        if jTrI == 0:
            with tf.compat.v1.Session() as sess:
                print(tf.compat.v1.count_nonzero(tmpTrImage - tmpTImage, axis=[0, 1]).eval())
            print(iTeI, jTrI, l2norm)
        arrayL2Norm = np.append(arrayL2Norm, l2norm)

    sIndex = np.argsort(arrayL2Norm)  # sorting distance and returning indices that achieves sort

    kLabels = trLabels[sIndex[0:paramk]]  # choose first k labels
    (values, counts) = np.unique(kLabels, return_counts=True)  # find unique labels and their counts
    arrayKNNLabels = np.append(arrayKNNLabels, values[np.argmax(counts)])

    if arrayKNNLabels[-1] != tLabels[iTeI]:
        numErrs += 1
        print(numErrs, "/", iTeI)
print("# Classification Errors= ", numErrs, "% accuracy= ", 100. * (numTestImages - numErrs) / numTestImages)
