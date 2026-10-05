import numpy as np
from core.Layer import Layer
class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        self.prev_y_pred = None
        self.prev_y_true = None

    def forward(self, y_pred, y_true):
        # Y pred and Y true are expected to be of shape (n_classes, n_samples)
        self.prev_y_pred = y_pred
        self.prev_y_true = y_true.reshape(y_pred.shape)  # Ensure y_true has the same shape as y_pred
        # Clip predictions to avoid log(0)
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(self.prev_y_true * np.log(y_pred_clipped) + (1 - self.prev_y_true) * np.log(1 - y_pred_clipped))

    def backward(self, out_gradient=1):
        n = self.prev_y_true.shape[1]  # Number of samples
        # Clip predictions to avoid division by zero
        epsilon = 1e-12
        y_pred_clipped = np.clip(self.prev_y_pred, epsilon, 1 - epsilon)
        return (1/n) * (-(self.prev_y_true / y_pred_clipped) + (1 - self.prev_y_true) / (1 - y_pred_clipped)) * out_gradient