from mnist import MNIST
import numpy as np 
import random

class Network:

  def __init__(self) -> None:
    self.W2 = np.random.randn(20, 784)
    self.W3 = np.random.randn(20, 20)
    self.W4 = np.random.randn(10, 20)

    self.b2 = np.random.randn(20, 1)
    self.b3  = np.random.randn(20, 1)
    self.b4 = np.random.randn(10, 1)

  def feed_forward(self, a1):
    z2 = self.sigmoid(np.dot(self.W2, a1) + self.b2)
    a2 = self.sigmoid(z2)

    z3 = np.dot(self.W3, a2) + self.b3
    a3 = self.sigmoid(z3)

    z4 = np.dot(self.W4, a3) + self.b4
    a4 = self.sigmoid(z4)
    return z2, z3, z4, a2, a3, a4

    
  def SGD(self, train_data: list[tuple[list[int], int]], batch_size: int, epochs: int, l_rate: float, 
          test_data: list[tuple[list[int], int]] = None):
    n = len(train_data)
    for i in range(epochs):
      random.shuffle(train_data)

      for j in range(n // batch_size):
        batch = train_data[j  * batch_size : (j + 1) * batch_size]
        x, y = zip(*batch)
        x = np.array(x).T
        y = self.one_hot_encode(y).T
        self.gradient_descent(x, l_rate, y)
      if test_data:
        self.evaluate(test_data)
        print(f"Epoch {i + 1}: {self.evaluate(test_data)}")


  def gradient_descent(self, x, l_rate: float, y):
    dw2, dw3, dw4, db2, db3, db4  = self.backprop(x, y)

    self.W2 -= l_rate * dw2  
    self.W3 -= l_rate * dw3
    self.W4 -= l_rate * dw4
    self.b2 -= l_rate * np.mean(db2, axis=1, keepdims=True)
    self.b3 -= l_rate * np.mean(db3, axis=1, keepdims=True)
    self.b4 -= l_rate * np.mean(db4, axis = 1, keepdims=True)
  

  def backprop(self, x, y):
    m = len(x)
    z2, z3, z4, a2, a3, a4 = self.feed_forward(x)
    delta_4 = (2 * (a4 - y)) * self.d_sigmoid(z4)
    db4 = 1/m * delta_4
    dw4 = 1/m * np.dot(delta_4, a3.T) 

    delta_3 =  self.W4.T @ delta_4 * self.d_sigmoid(z3)
    db3 = 1/m * delta_3
    dw3 = 1/m * np.dot(delta_3, a2.T) 
      
    delta_2 =  self.W3.T @ delta_3 * self.d_sigmoid(z2)
    db2 = 1/m * delta_2
    dw2 = 1/m * np.dot(delta_2, x.T)

    return dw2, dw3, dw4, db2, db3, db4 

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def d_sigmoid(self, x): 
    return self.sigmoid(x) * (1 - self.sigmoid(x)) 
  
  def one_hot_encode(self, y):
    encoded = np.zeros((len(y), 10))
    for i in range(len(y)):
      encoded[i][y[i]] = 1
    return encoded
  
  def evaluate(self, test_data: list[tuple[list[int], int]]):
    x, y = zip(*test_data)
    x = np.array(x).T
    y = self.one_hot_encode(y).T
    _, _, _, _, _, a4 = self.feed_forward(x)
    predictions = np.argmax(a4, axis=0)  # Get the indices of maximum values along columns

    accuracy = np.mean(predictions == y)  # Compare predictions with true labels
    return accuracy
    

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
print(type(train_data[0][0]))

nn = Network()
nn.SGD(train_data, 10, 10, 2, test_data)
 
