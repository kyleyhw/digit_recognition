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


class Flatten(Layer):
    """
    Flattens the input into a 2D array (batch_size, num_features).
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        batch_size = input.shape[0]
        # Flatten all dimensions except the batch dimension
        self.output = input.reshape(batch_size, -1)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Reshape the output_gradient back to the original input shape
        return output_gradient.reshape(self.input_shape)


class Convolutional(Layer):
    """
    A convolutional layer for neural networks.
    Performs convolution operation using im2col for efficiency.
    Includes a naive looping implementation for educational purposes.
    """
    def __init__(self, input_shape, num_filters, kernel_size, stride=(1, 1), padding='valid'):
        super().__init__()
        self.input_shape = input_shape # (in_height, in_width, in_channels)
        self.num_filters = num_filters
        self.kernel_size = kernel_size # (k_height, k_width)
        self.stride = stride # (s_height, s_width)
        self.padding = padding

        in_height, in_width, in_channels = input_shape
        k_height, k_width = kernel_size
        s_height, s_width = stride

        # Calculate output dimensions
        if padding == 'same':
            self.pad_h = (s_height * (in_height - 1) + k_height - in_height) // 2
            self.pad_w = (s_width * (in_width - 1) + k_width - in_width) // 2
            self.out_height = (in_height + 2 * self.pad_h - k_height) // s_height + 1
            self.out_width = (in_width + 2 * self.pad_w - k_width) // s_width + 1
        elif padding == 'valid':
            self.pad_h = 0
            self.pad_w = 0
            self.out_height = (in_height - k_height) // s_height + 1
            self.out_width = (in_width - k_width) // s_width + 1
        else:
            raise ValueError("Padding must be 'valid' or 'same'")

        # Initialize weights (filters) using Xavier initialization
        # Shape: (k_height, k_width, in_channels, num_filters)
        self.weights = np.random.randn(k_height, k_width, in_channels, num_filters) * np.sqrt(2.0 / (k_height * k_width * in_channels))
        self.bias = np.zeros(num_filters)

        # Store for backward pass
        self.im2col_matrix = None
        self.input_padded = None

    def forward(self, input):
        self.input = input
        batch_size, in_height, in_width, in_channels = input.shape
        k_height, k_width = self.kernel_size
        s_height, s_width = self.stride

        # Pad input
        self.input_padded = self._pad_input(input)
        padded_height, padded_width = self.input_padded.shape[1:3]

        # Reshape weights for matrix multiplication
        weights_reshaped = self.weights.reshape(-1, self.num_filters) # (k_height * k_width * in_channels, num_filters)

        # Convert input to im2col matrix
        self.im2col_matrix = self._im2col(self.input_padded)

        # Perform matrix multiplication
        output_im2col = np.dot(self.im2col_matrix, weights_reshaped)

        # Reshape output back to (batch_size, out_height, out_width, num_filters)
        self.output = output_im2col.reshape(batch_size, self.out_height, self.out_width, self.num_filters) + self.bias

        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, in_height, in_width, in_channels = self.input.shape
        k_height, k_width = self.kernel_size
        s_height, s_width = self.stride

        # Reshape output_gradient to (batch_size * out_height * out_width, num_filters)
        output_gradient_reshaped = output_gradient.reshape(-1, self.num_filters)

        # Reshape weights for matrix multiplication
        weights_reshaped = self.weights.reshape(-1, self.num_filters)

        # Gradient with respect to weights
        self.weights_gradient = np.dot(self.im2col_matrix.T, output_gradient_reshaped)
        self.weights_gradient = self.weights_gradient.reshape(self.weights.shape) # Reshape back to original weights shape

        # Gradient with respect to bias
        self.bias_gradient = np.sum(output_gradient, axis=(0, 1, 2))

        # Gradient with respect to input (im2col format)
        input_gradient_im2col = np.dot(output_gradient_reshaped, weights_reshaped.T)

        # Convert input_gradient_im2col back to original input shape
        input_gradient_padded = self._col2im(input_gradient_im2col, self.input_padded.shape)

        # Remove padding from input_gradient if padding was applied in forward pass
        if self.padding == 'same':
            pad_h, pad_w = self._get_padding_dims()
            input_gradient = input_gradient_padded[:, pad_h:pad_h+in_height, pad_w:pad_w+in_width, :]
        else:
            input_gradient = input_gradient_padded

        # Update weights and bias
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient

        return input_gradient

    # --- Naive Looping Implementation (for educational purposes) ---
    def _naive_convolve_forward(self, input_data):
        batch_size, in_height, in_width, in_channels = input_data.shape
        k_height, k_width = self.kernel_size
        s_height, s_width = self.stride

        # Pad input
        input_padded = self._pad_input(input_data)
        padded_height, padded_width = input_padded.shape[1:3]

        output = np.zeros((batch_size, self.out_height, self.out_width, self.num_filters))

        for b in range(batch_size):
            for h in range(self.out_height):
                for w in range(self.out_width):
                    for f in range(self.num_filters):
                        h_start = h * s_height
                        h_end = h_start + k_height
                        w_start = w * s_width
                        w_end = w_start + k_width

                        # Extract receptive field
                        patch = input_padded[b, h_start:h_end, w_start:w_end, :]
                        
                        # Perform element-wise multiplication and sum
                        output[b, h, w, f] = np.sum(patch * self.weights[:, :, :, f]) + self.bias[f]
        return output

    def _naive_convolve_backward(self, output_gradient, learning_rate):
        batch_size, in_height, in_width, in_channels = self.input.shape
        k_height, k_width = self.kernel_size
        s_height, s_width = self.stride

        # Pad input
        input_padded = self._pad_input(self.input)
        padded_height, padded_width = input_padded.shape[1:3]

        # Initialize gradients
        input_gradient_padded = np.zeros_like(input_padded)
        weights_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)

        for b in range(batch_size):
            for h in range(self.out_height):
                for w in range(self.out_width):
                    for f in range(self.num_filters):
                        h_start = h * s_height
                        h_end = h_start + k_height
                        w_start = w * s_width
                        w_end = w_start + k_width

                        # Extract receptive field
                        patch = input_padded[b, h_start:h_end, w_start:w_end, :]

                        # Gradient with respect to bias
                        bias_gradient[f] += output_gradient[b, h, w, f]

                        # Gradient with respect to weights
                        weights_gradient[:, :, :, f] += patch * output_gradient[b, h, w, f]

                        # Gradient with respect to input
                        input_gradient_padded[b, h_start:h_end, w_start:w_end, :] += self.weights[:, :, :, f] * output_gradient[b, h, w, f]
        
        # Remove padding from input_gradient if padding was applied in forward pass
        if self.padding == 'same':
            pad_h, pad_w = self._get_padding_dims()
            input_gradient = input_gradient_padded[:, pad_h:pad_h+in_height, pad_w:pad_w+in_width, :]
        else:
            input_gradient = input_gradient_padded

        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        self.weights_gradient = weights_gradient # Store for gradient check
        self.bias_gradient = bias_gradient # Store for gradient check

        return input_gradient

    def _get_padding_dims(self):
        return self.pad_h, self.pad_w

    def _pad_input(self, input_data):
        if self.padding == 'valid':
            return input_data
        pad_h, pad_w = self._get_padding_dims()
        return np.pad(input_data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    def _im2col(self, input_data):
        """
        Converts input data into a 2D matrix where each column is a flattened receptive field.
        Input: (batch_size, in_height, in_width, in_channels)
        Output: (out_height * out_width * batch_size, k_height * k_width * in_channels)
        """
        batch_size, in_height, in_width, in_channels = input_data.shape
        k_height, k_width = self.kernel_size
        s_height, s_width = self.stride

        # Calculate output dimensions (already done in __init__, but useful for clarity)
        out_height = (in_height - k_height) // s_height + 1
        out_width = (in_width - k_width) // s_width + 1

        # Allocate memory for the im2col matrix
        im2col_matrix = np.zeros((batch_size * out_height * out_width, k_height * k_width * in_channels))

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * s_height
                    h_end = h_start + k_height
                    w_start = w * s_width
                    w_end = w_start + k_width

                    # Extract the receptive field
                    patch = input_data[b, h_start:h_end, w_start:w_end, :].reshape(-1)
                    
                    # Place the patch into the im2col matrix
                    row_idx = b * out_height * out_width + h * out_width + w
                    im2col_matrix[row_idx, :] = patch
        
        return im2col_matrix

    def _col2im(self, col_matrix, input_shape_for_col2im):
        """
        Converts a 2D column matrix back to the original input shape.
        Used in backward pass to distribute gradients.
        """
        batch_size, in_height, in_width, in_channels = input_shape_for_col2im
        k_height, k_width = self.kernel_size
        s_height, s_width = self.stride

        # Calculate output dimensions (already done in __init__)
        out_height = (in_height - k_height) // s_height + 1
        out_width = (in_width - k_width) // s_width + 1

        # Allocate memory for the gradient with respect to input
        input_gradient = np.zeros(input_shape_for_col2im)

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * s_height
                    h_end = h_start + k_height
                    w_start = w * s_width
                    w_end = w_start + k_width

                    # Extract the gradient patch from the col_matrix
                    row_idx = b * out_height * out_width + h * out_width + w
                    patch_gradient = col_matrix[row_idx, :].reshape(k_height, k_width, in_channels)
                    
                                                                                # Add the gradient patch to the input_gradient
                    input_gradient[b, h_start:h_end, w_start:w_end, :] += patch_gradient
        
        return input_gradient


class MaxPooling(Layer):
    """
    Max Pooling layer.
    Reduces the spatial dimensions of the input.
    """
    def __init__(self, pool_size, stride=(1, 1)):
        super().__init__()
        self.pool_size = pool_size # (p_height, p_width)
        self.stride = stride # (s_height, s_width)
        self.input_shape = None
        self.output_shape = None
        self.max_indices = None # Store indices of max values for backward pass

    def forward(self, input):
        self.input = input
        batch_size, in_height, in_width, in_channels = input.shape
        p_height, p_width = self.pool_size
        s_height, s_width = self.stride

        out_height = (in_height - p_height) // s_height + 1
        out_width = (in_width - p_width) // s_width + 1

        self.output_shape = (batch_size, out_height, out_width, in_channels)
        self.output = np.zeros(self.output_shape)
        self.max_indices = np.zeros(self.output_shape, dtype=object) # Store indices as objects to handle tuples

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(in_channels):
                        h_start = h * s_height
                        h_end = h_start + p_height
                        w_start = w * s_width
                        w_end = w_start + p_width

                        # Extract the receptive field
                        patch = input[b, h_start:h_end, w_start:w_end, c]
                        
                        # Find the maximum value and its index
                        max_value = np.max(patch)
                        max_idx_flat = np.argmax(patch)
                        
                        # Convert flattened index to 2D relative index within the patch
                        max_h_rel = max_idx_flat // p_width
                        max_w_rel = max_idx_flat % p_width

                        self.output[b, h, w, c] = max_value
                        self.max_indices[b, h, w, c] = (max_h_rel, max_w_rel) # Store relative (h,w) index

        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, in_height, in_width, in_channels = self.input.shape
        p_height, p_width = self.pool_size
        s_height, s_width = self.stride

        input_gradient = np.zeros_like(self.input)

        for b in range(batch_size):
            for h in range(self.output_shape[1]):
                for w in range(self.output_shape[2]):
                    for c in range(in_channels):
                        h_start = h * s_height
                        w_start = w * s_width

                        # Get the stored relative index of the max value
                        max_h_rel, max_w_rel = self.max_indices[b, h, w, c]

                        # Propagate gradient only to the max element
                        input_gradient[b, h_start + max_h_rel, w_start + max_w_rel, c] += output_gradient[b, h, w, c]
        
        return input_gradient
