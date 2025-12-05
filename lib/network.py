class Network:
    """Container class: holds layers, performs forward/backward passes, training."""

    def __init__(self, layers):
        self.layers = layers
        self.loss_history = []

    def forward(self, x):
        # Pass input through each layer
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        # Backprop: go in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X, y, loss_fn, optimizer, epochs=1000, verbose=True):

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            loss = loss_fn.loss(y_pred, y)
            self.loss_history.append(loss)

            # Backward pass
            grad = loss_fn.grad(y_pred, y)
            self.backward(grad)

            # Weight update
            optimizer.step(self.layers)

            # Print every 500 epochs
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def summary(self):
        """Print architecture and shapes of learned parameters."""
        print("\nNetwork Architecture:")
        print("=" * 50)
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__

            if hasattr(layer, "W"):
                print(f"Layer {i}: {layer_name} | Weights: {layer.W.shape} | Biases: {layer.b.shape}")
            else:
                print(f"Layer {i}: {layer_name}")
        print("=" * 50)
