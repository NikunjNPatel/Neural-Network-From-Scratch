class SGD:
    """Update parameters using their most recently computed gradients."""

    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        """Apply parameter -= learning_rate * gradient in place."""
        for parameter in self.model.parameters():
            gradient = parameter["gradient"]
            if gradient is None:
                raise ValueError(f"No gradient for {parameter['name']}; call backward() first")
            parameter["value"] -= self.lr * gradient
