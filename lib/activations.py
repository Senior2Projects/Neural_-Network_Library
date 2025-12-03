import numpy as np
from .layers import Layer

class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)

class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)

class ReLU(Layer):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (self.out > 0)

class Softmax(Layer):
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # To avoid overflow and keep computation safe
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True) # axis=1 → compute max per row, keepdims=True → keeps the output as a column vector instead of flattening it
        return self.out

    def backward(self, grad_output):
        return grad_output