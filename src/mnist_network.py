from mnist import MNIST
import numpy as np 

class Network:

  def __init__(self) -> None:
    self.W1 = np.random.randn(20, 784)
    self.W2 = np.random.randn(20, 20)
    self.W3 = np.random.randn(10, 20)

    self.b1 = np.random.randn(784, 1)
    self.b2  = np.random.randn(20, 1)
    self.b3 = np.random.randn(20, 1)

  def feed_forward(self, x):
    z1 = self.sigmoid(np.dot(self.W1, x) + self.b1)
    a1 = self.sigmoid(z1)

    z2 = np.dot(self.W2, a1) + self.b2
    a2 = self.sigmoid(z2)

    z3 = np.dot(self.W3, a2) + self.b3
    a3 = self.sigmoid(z3)

    return a3




  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))



mndata = MNIST('./data')
images, labels = mndata.load_training()

images = np.array(images)
labels = np.array(labels)
#print(images.shape)
#print(labels.shape)
 
