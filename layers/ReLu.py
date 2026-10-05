from core.Layer import Layer
import numpy as np

class ReLu(Layer):

    def __init__(self):
        self.prev_X = None

    def forward(self, X):
        self.prev_X = X
        return np.maximum(0, X)

    def backward(self, out_gradient):
        relu_grad = np.where(self.prev_X > 0, 1, 0)
        return out_gradient * relu_grad