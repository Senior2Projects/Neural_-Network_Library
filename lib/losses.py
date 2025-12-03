import numpy as np

class MSE:
    @staticmethod #can be called without creating an instance of the class
    def loss(y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def grad(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]