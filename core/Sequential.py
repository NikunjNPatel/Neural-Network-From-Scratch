class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        X = X.T  # Transpose the input to match the expected shape (features, samples)
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, out_gradient):
        for layer in reversed(self.layers):
            out_gradient = layer.backward(out_gradient)
        return out_gradient

    def parameters(self):
        """Return the trainable parameters from all layers."""
        return [parameter for layer in self.layers for parameter in layer.parameters()]
