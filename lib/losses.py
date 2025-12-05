import numpy as np

class MSE:
    """Mean Squared Error loss function."""

    @staticmethod
    def loss(y_pred, y_true):
        # Average squared difference
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def grad(y_pred, y_true):
        # Derivative: 2*(pred - true)/N
        return 2 * (y_pred - y_true) / y_true.shape[0]
