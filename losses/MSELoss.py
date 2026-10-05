from core.Layer import Layer
import numpy as np
class MSELoss(Layer):
 
    def __init__(self):
        self.prev_y_pred = None
        self.prev_y_true = None

    def forward(self, y_pred, y_true):
        # Y pred and Y true are expected to be of shape (n_classes, n_samples)
        self.prev_y_pred = y_pred
        self.prev_y_true = y_true.reshape(y_pred.shape)  # Ensure y_true has the same shape as y_pred
        return np.mean((self.prev_y_pred - self.prev_y_true) ** 2)

    def backward(self, out_gradient=1):
        n = self.prev_y_true.shape[1] # Number of samples
        return (2 / n) * (self.prev_y_pred - self.prev_y_true) * out_gradient