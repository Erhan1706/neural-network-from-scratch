from mnist import MNIST
import numpy as np 
import random
from scipy.special import expit

"""
Network class that implements a simple neural network hardcoded with 2 hidden layers. 
Each hidden layer has 20 neurons. The network uses the sigmoid activation function.
"""
class Network:

  def __init__(self) -> None:
    self.W2 = np.random.randn(20, 784)
    self.W3 = np.random.randn(20, 20)
    self.W4 = np.random.randn(10, 20)

    self.b2 = np.random.randn(20, 1)
    self.b3  = np.random.randn(20, 1)
    self.b4 = np.random.randn(10, 1)

  def feed_forward(self, a1: list[float]):
    """
    Feed forward the input through the network and return the activations and the weighted sums of the neurons.
    First activation layer should be of shape (784,1).
    """
    z2 = self.sigmoid(np.dot(self.W2, a1) + self.b2)
    a2 = self.sigmoid(z2)

    z3 = np.dot(self.W3, a2) + self.b3
    a3 = self.sigmoid(z3)

    z4 = np.dot(self.W4, a3) + self.b4
    a4 = self.sigmoid(z4)
    return z2, z3, z4, a2, a3, a4

  def SGD(self, train_data: list[tuple[list[float], int]], batch_size: int, epochs: int, l_rate: float, 
          test_data: list[tuple[list[float], int]] = None):
    n = len(train_data)
    for i in range(epochs):
      random.shuffle(train_data)

      for j in range(n // batch_size):
        batch = train_data[j  * batch_size : (j + 1) * batch_size]
        x, y = zip(*batch)
        x = np.array(x).T
        y = self.one_hot_encode(y) # This cannot be transposed
        self.gradient_descent(x, l_rate, y, batch_size)
      if test_data:
        self.evaluate(test_data)
        print(f"Epoch {i + 1}: {self.evaluate(test_data)}/{len(test_data)}")


  def gradient_descent(self, x: list[list[float]], l_rate: float, y, batch_size: int):
    dw2, dw3, dw4, db2, db3, db4  = self.backprop(x, y)

    l_rate = l_rate/batch_size 
    self.W2 -= l_rate * dw2  
    self.W3 -= l_rate * dw3
    self.W4 -= l_rate * dw4
    self.b2 -= l_rate * np.mean(db2, axis=1, keepdims=True)
    self.b3 -= l_rate * np.mean(db3, axis=1, keepdims=True)
    self.b4 -= l_rate * np.mean(db4, axis=1, keepdims=True)
  

  def backprop(self, x:list[list[float]], y:list[list[int]]):
    z2, z3, z4, a2, a3, a4 = self.feed_forward(x)
    delta_4 = (2*(a4 - y.T)) * self.d_sigmoid(z4)
    db4 = 1 * delta_4
    dw4 = 1 * np.dot(delta_4, a3.T) 

    delta_3 =  self.W4.T @ delta_4 * self.d_sigmoid(z3)
    db3 = 1 * delta_3
    dw3 = 1 * np.dot(delta_3, a2.T) 
      
    delta_2 =  self.W3.T @ delta_3 * self.d_sigmoid(z2)
    db2 = 1 * delta_2
    dw2 = 1 * np.dot(delta_2, x.T)

    return dw2, dw3, dw4, db2, db3, db4 

  def sigmoid(self, x:list[list[float]]):
    return expit(x) 
  
  def d_sigmoid(self, x:list[list[float]]): 
    return self.sigmoid(x) * (1 - self.sigmoid(x)) 
  
  def one_hot_encode(self, y: tuple[int]):
    encoded = np.zeros((len(y), 10))
    for i in range(len(y)):
      encoded[i][y[i]] = 1
    return encoded
  
  def evaluate(self, test_data: list[tuple[list[list[float]], int]]):
    test_results = [(np.argmax(self.feed_forward(x.reshape((784,1)))[5]), y)
                        for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)
    

mndata = MNIST('./data')
images, labels = mndata.load_training()
images = np.array(images)
labels = np.array(labels)

images_train = images[:50000]
images_test = images[50000:]
labels_train = labels[:50000]
labels_test = labels[50000:]

train_data = list(zip(images_train, labels_train))
test_data = list(zip(images_test, labels_test))

nn = Network()
nn.SGD(train_data, 10, 10, 2.0, test_data)
 
