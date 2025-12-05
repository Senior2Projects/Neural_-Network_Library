import numpy as np

class Layer:
    """Base class — every layer must implement forward & backward."""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Dense(Layer):
    """Fully connected linear layer: output = xW + b"""

    def __init__(self, input_size, output_size):
        # Random weight initialization (small)
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        # Store input for gradient computation later
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        # Compute gradients of weights and biases
        self.dW = self.x.T @ grad_output # shape: (input_size, output_size)
        self.db = np.sum(grad_output, axis=0, keepdims=True) # shape: (1, output_size)

        # Pass gradient backward to previous layer
        return grad_output @ self.W.T