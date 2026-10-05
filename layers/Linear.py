from core.Layer import Layer
import numpy as np
class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size)*np.sqrt(1/in_size)
        self.b = np.zeros((out_size,1))
        self.dW = None
        self.db = None
        self.prev_X = None

    def forward(self, X):
        self.prev_X = X
        return self.W.T @ X + self.b

    def backward(self, out_gradient):
        self.dW = self.prev_X @ out_gradient.T
        self.db = np.sum(out_gradient, axis=1, keepdims=True)
        return self.W @ out_gradient # Gradient of loss with respect to input

    def parameters(self):
        return [
            {
                "name": "W",
                "value": self.W,
                "gradient": self.dW
            },
            {
                "name": "b",
                "value": self.b,
                "gradient": self.db
            }
        ]