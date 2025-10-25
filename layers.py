import numpy as np

class Layer:
    """
    Base class for neural network layers.
    All layers should inherit from this class and implement the forward and backward methods.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Performs the forward pass through the layer.
        Should store the input for use in the backward pass.
        Returns the output of the layer.
        """
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """
        Performs the backward pass through the layer.
        Computes the gradient of the loss with respect to the input of the layer
        and updates any layer parameters.
        Returns the gradient of the loss with respect to the input.
        """
        raise NotImplementedError
