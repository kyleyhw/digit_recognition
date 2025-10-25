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


class Dense(Layer):
    """
    A fully connected neural network layer.
    Performs a linear transformation: output = input @ W + b
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Gradient with respect to weights
        self.weights_gradient = np.dot(self.input.T, output_gradient) # Store for gradient check
        # Gradient with respect to bias
        self.bias_gradient = np.sum(output_gradient, axis=0) # Store for gradient check
        # Gradient with respect to input
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update weights and bias
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient

        return input_gradient


class Activation(Layer):
    """
    Base class for activation layers.
    Subclasses must implement the activation and its derivative.
    """
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation layer.
    f(x) = max(0, x)
    """
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    """
    Sigmoid activation layer.
    f(x) = 1 / (1 + e^(-x))
    """
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    """
    Softmax activation layer.
    Converts a vector of arbitrary real values into a probability distribution.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        # Subtract max for numerical stability
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Softmax derivative is more complex.
        # For cross-entropy loss, dL/d(input) = output - target.
        # Here, we assume output_gradient is dL/d(softmax_output).
        # The Jacobian matrix of softmax is S_ij = s_i * (delta_ij - s_j)
        # dL/d(input)_j = sum_i (dL/d(output)_i * d(output)_i / d(input)_j)
        # This simplifies to:
        s = self.output
        return s * output_gradient - s * np.sum(s * output_gradient, axis=-1, keepdims=True) 
