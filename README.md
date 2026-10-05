# Neural-Network-From-Scratch

## Stochastic gradient descent

`SGD` updates each parameter after a forward and backward pass. To make it
*stochastic*, feed one randomly chosen training example into each pass:

```python
import numpy as np

from core.Sequential import Sequential
from layers.Linear import Linear
from losses.MSELoss import MSELoss
from optimizers.SGD import SGD

# Each column is one example: X has shape (features, examples).
X = np.array([[-1.0, 0.0, 1.0]])
y = 2 * X + 3

model = Sequential([Linear(1, 1)])
loss_function = MSELoss()
optimizer = SGD(model, learning_rate=0.1)

for epoch in range(100):
    for i in np.random.permutation(X.shape[1]):
        prediction = model.forward(X[:, i:i + 1])
        loss_function.forward(prediction, y[:, i:i + 1])
        model.backward(loss_function.backward())
        optimizer.step()

print(model.layers[0].W, model.layers[0].b)
```

The backward pass in this project replaces each layer's gradients on every
call, so there is no separate gradient clearing step.
