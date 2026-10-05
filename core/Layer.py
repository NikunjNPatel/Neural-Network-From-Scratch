class Layer:
    def __init__(self):
        None

    def forward(self, X):
        return NotImplementedError

    def backward(self, gradient):
        return NotImplementedError

    def parameters(self):
        return []