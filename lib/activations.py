import numpy as np
from .layers import Layer

class Sigmoid(Layer):
    # Forward pass: standard sigmoid activation
    def forward(self, x):
        self.out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )
        return self.out

    # Backward pass: derivative of sigmoid = s*(1-s)
    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)


class Tanh(Layer):
    # Forward pass
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    # Backward pass: derivative = 1 - tanh(x)^2
    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)


class ReLU(Layer):
    # Forward: max(0, x)
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out

    # Backward: gradient passes only where x > 0
    def backward(self, grad_output):
        return grad_output * (self.out > 0)


class Softmax(Layer):
    # Softmax with numerical stability
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.out

    # Backward: used for MSE or external derivative
    # (true Jacobian not computed here)
    def backward(self, grad_output):
        return grad_output
