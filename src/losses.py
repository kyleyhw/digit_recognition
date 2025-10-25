import numpy as np

class Loss:
    """
    Base class for loss functions.
    Subclasses must implement the loss calculation and its derivative.
    """
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def prime(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    L = 1/N * sum((y_pred - y_true)^2)
    """
    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_pred - y_true, 2))

    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss function.
    Used for multi-class classification.
    L = -1/N * sum(y_true * log(y_pred))
    """
    def loss(self, y_true, y_pred):
        # Clip y_pred to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def prime(self, y_true, y_pred):
        # Clip y_pred to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        # Derivative for Softmax + Cross-Entropy is simply y_pred - y_true
        # However, if we want the derivative of Cross-Entropy alone, it's:
        return -y_true / y_pred / y_true.shape[0]
