from mnist import MNIST
import numpy as np 

mndata = MNIST('./data')
images, labels = mndata.load_training()

images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)

