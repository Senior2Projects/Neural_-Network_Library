import numpy as np

class Layer:
    """Base class for all layers."""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer."""
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x  # Store input for backward
        return x @ self.W + self.b

    def backward(self, grad_output):
        # Gradients of weights and biases
        self.dW = self.x.T @ grad_output  # shape: (input_size, output_size)
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # shape: (1, output_size)
        # Gradient to pass to previous layer
        return grad_output @ self.W.T