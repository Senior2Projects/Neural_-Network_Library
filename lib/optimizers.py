class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, layers):
        # Update every layer that has weights
        for layer in layers:
            if hasattr(layer, 'W'):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db