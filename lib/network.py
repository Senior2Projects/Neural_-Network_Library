import numpy as np
class Network:
    """Container class: holds layers, performs forward/backward passes, training."""

    def __init__(self, layers):
        self.layers = layers
        self.loss_history = []

    def save_weights(self, filename="model_weights.npz"):
        """Saves weights (W) and biases (b) of all Dense layers to a single .npz file."""
        weight_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W") and hasattr(layer, "b"):
                # Use a unique key for each weight/bias (e.g., 'L0_W', 'L0_b', 'L2_W', etc.)
                weight_dict[f"L{i}_W"] = layer.W
                weight_dict[f"L{i}_b"] = layer.b
        
        # np.savez saves all keyword arguments as individual arrays in a single file
        np.savez(filename, **weight_dict)
        print(f"✅ Weights saved to {filename}")

    def load_weights(self, filename="model_weights.npz"):
        """Loads weights (W) and biases (b) from a .npz file into all Dense layers."""
        try:
            loaded_data = np.load(filename)
        except FileNotFoundError:
            print(f"❌ Error: Weights file not found at {filename}")
            return self

        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W") and hasattr(layer, "b"):
                w_key = f"L{i}_W"
                b_key = f"L{i}_b"
                
                if w_key in loaded_data and b_key in loaded_data:
                    # Check shapes before loading to prevent runtime errors
                    if layer.W.shape == loaded_data[w_key].shape and layer.b.shape == loaded_data[b_key].shape:
                        layer.W = loaded_data[w_key]
                        layer.b = loaded_data[b_key]
                        # print(f"Loaded weights for Layer {i}")
                    else:
                        print(f"❌ Warning: Shape mismatch for Layer {i}. Skipped loading for this layer.")
                else:
                    print(f"❌ Warning: Missing keys ({w_key} or {b_key}) in file for Layer {i}. Skipping.")

        print(f"✅ Weights loaded from {filename}")
        return self

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
                #i want to see the epoch no 1000 loss too
        if verbose:
            print(f"Epoch {epochs-1}, Loss: {loss:.6f}")
            

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

    def print_final_predictions(self, X, y):
        """Prints input target prediction table after training."""
        print("\nFinal model predictions:")
        print("=" * 60)

        preds = self.forward(X)

        print(f"{'Input (x1,x2)':<20} {'Target':<10} {'Predicted':<12} {'Error':<10}")
        print("-" * 60)

        for i in range(len(X)):
            x1, x2 = X[i]
            target = y[i][0]
            pred = preds[i][0]
            error = abs(target - pred)

            status = "✓" if error < 0.1 else "✗"

            print(f"({x1:>2}, {x2:>2})".ljust(20),
                f"{target:+.4f}".ljust(10),
                f"{pred:+.4f}".ljust(12),
                f"{error:.4f}  {status}")

        print("-" * 60)
    