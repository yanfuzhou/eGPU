# Reference: https://colab.research.google.com/github/FreeOfConfines/ExampleNNWithKerasAndTensorflow/blob/master/K_Nearest_Neighbor_Classification_with_Tensorflow_on_Fashion_MNIST_Dataset.ipynb#scrollTo=WEy4WAsBzFVo
# Placeholder are not supported by eager: https://medium.com/coinmonks/8-things-to-do-differently-in-tensorflows-eager-execution-mode-47cf429aa3ad
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

paramk = 11  # parameter k of K-nearest neighbors

# Defining KNN Graph with L0 Norm
x = tf.compat.v1.placeholder(trImages.dtype, shape=trImages.shape)  # all train images, i.e., 60000 x 28 x 28
y = tf.compat.v1.placeholder(tImages.dtype, shape=tImages.shape[1:])  # a test image, 28 x 28

xThresholded = tf.compat.v1.clip_by_value(tf.compat.v1.cast(x, tf.int32), 0,
                                1)  # x is int8 which is not supported in many tf functions, hence typecast
yThresholded = tf.compat.v1.clip_by_value(tf.compat.v1.cast(y, tf.int32), 0,
                                1)  # clip_by_value converts dataset to tensors of 0 and 1, i.e., 1 where tensor is non-zero
computeL0Dist = tf.compat.v1.count_nonzero(xThresholded - yThresholded, axis=[1, 2])  # Computing L0 Norm by reducing along axes
findKClosestTrImages = tf.contrib.framework.argsort(computeL0Dist,
                                                    direction='ASCENDING')  # sorting (image) indices in order of ascending metrics, pick first k in the next step
findLabelsKClosestTrImages = tf.gather(trLabels, findKClosestTrImages[
                                                 0:paramk])  # doing trLabels[findKClosestTrImages[0:paramk]] throws error, hence this workaround
findULabels, findIdex, findCounts = tf.unique_with_counts(
    findLabelsKClosestTrImages)  # examine labels of k closest Train images
findPredictedLabel = tf.gather(findULabels, tf.argmax(
    findCounts))  # assign label to test image based on most occurring labels among k closest Train images

# Let's run the graph
numErrs = 0
numTestImages = np.shape(tLabels)[0]
numTrainImages = np.shape(trLabels)[0]  # so many train images

with tf.Session() as sess:
    for iTeI in range(0, numTestImages):  # iterate each image in test set
        predictedLabel = sess.run([findPredictedLabel], feed_dict={x: trImages, y: tImages[iTeI]})

        if predictedLabel == tLabels[iTeI]:
            numErrs += 1
            print(numErrs, "/", iTeI)
            print("\t\t", predictedLabel[0], "\t\t\t\t", tLabels[iTeI])

            if (1):
                plt.figure(1)
                plt.subplot(1, 2, 1)
                plt.imshow(tImages[iTeI])
                plt.title('Test Image has label %i' % (predictedLabel[0]))

                for i in range(numTrainImages):
                    if trLabels[i] == predictedLabel:
                        plt.subplot(1, 2, 2)
                        plt.imshow(trImages[i])
                        plt.title('Correctly Labeled as %i' % (tLabels[iTeI]))
                        plt.draw()
                        break
                plt.show()

print("# Classification Errors= ", numErrs, "% accuracy= ", 100. * (numTestImages - numErrs) / numTestImages)
