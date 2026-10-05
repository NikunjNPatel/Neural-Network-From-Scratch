from core.Layer import Layer
import numpy as np

class Sigmoid(Layer):

    def __init__(self):
        self.prev_X = None

    def forward(self, X):
        self.prev_X = X
        return 1 / (1 + np.exp(-X))

    def backward(self, out_gradient):
        sigmoid_grad = self.forward(self.prev_X) * (1 - self.forward(self.prev_X))
        return out_gradient * sigmoid_grad