# Deep Dive: The Layers of a Neural Network

This document provides a detailed, beginner-friendly explanation of each layer implemented in `src/layers.py`. We will explore the purpose of each layer, the mathematics behind its operation, and how those mathematics are reflected in the code.

---

## The Base `Layer` Class

All layers in our network inherit from a simple base class that establishes a common interface.

*   **Purpose:** To ensure that every component we build can be treated polymorphically by our `Network` class. It guarantees that every layer will have a `forward` and a `backward` method.

*   **Code (`src/layers.py`):
    ```python
    class Layer:
        def __init__(self):
            self.input = None
            self.output = None

        def forward(self, input):
            raise NotImplementedError

        def backward(self, output_gradient, learning_rate):
            raise NotImplementedError
    ```

---

## `Dense` Layer (Fully Connected Layer)

*   **What it is:** The most basic and common type of layer. Each neuron in a dense layer receives input from *all* the neurons in the previous layer, hence the name "fully connected."

*   **The Math:** It performs a linear transformation on the input data.
    *   **Forward Pass:** $Z = XW + b$
    *   **Backward Pass (Gradients):**
        *   Gradient w.r.t. Weights ($dW$): $dW = X^T \]cdot dZ$
        *   Gradient w.r.t. Biases ($db$): $db = \sum(dZ)$
        *   Gradient w.r.t. Input ($dX$): $dX = dZ \cdot W^T$

*   **The Code (`src/layers.py`):

    **Initialization:**
    ```python
    class Dense(Layer):
        def __init__(self, input_size, output_size):
            # ...
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            self.bias = np.zeros(output_size)
    ```
    We initialize the weights ($W$) with small random numbers (Xavier initialization) to break symmetry and help with training. The biases ($b$) are initialized to zero.

    **Forward Pass:**
    ```python
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    ```
    This is a direct translation of $Z = XW + b$ using NumPy's dot product for the matrix multiplication $XW$.

    **Backward Pass:**
    ```python
    def backward(self, output_gradient, learning_rate):
        # dW = X^T * dZ
        weights_gradient = np.dot(self.input.T, output_gradient)
        # db = sum(dZ)
        bias_gradient = np.sum(output_gradient, axis=0)
        # dX = dZ * W^T
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update parameters using Gradient Descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
    ```
    Here, `output_gradient` is $dZ$. The code calculates the gradients for the weights and biases and then updates them using the learning rate. Finally, it calculates and returns the gradient with respect to its input, to be passed to the previous layer.

---

## `ReLU` Activation Layer

*   **What it is:** A non-linear activation function. It acts as a simple switch: if the input is positive, it passes it through; if it's negative, it outputs zero.

*   **The Math:**
    *   **Forward Pass:** $f(x) = \max(0, x)$
    *   **Backward Pass (Derivative):**
        $$ f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases} $$

*   **The Code (`src/layers.py`):
    ```python
    class ReLU(Activation):
        def __init__(self):
            def relu(x):
                return np.maximum(0, x)

            def relu_prime(x):
                return (x > 0).astype(float)

            super().__init__(relu, relu_prime)
    ```
    The `ReLU` class simply provides the specific `relu` and `relu_prime` functions to its parent `Activation` class, which handles the generic forward/backward logic for all activation functions.

---

## `Convolutional` Layer

*   **What it is:** The core of a CNN. Instead of looking at all inputs at once, it uses small, learnable **filters** (or **kernels**) that slide across the input image to detect specific features like edges, corners, or textures.

*   **The Math:** The forward pass is a cross-correlation operation. The backward pass is a more complex full convolution. For efficiency, we use the `im2col` (image-to-column) technique, which transforms this complex operation into a single, highly optimized matrix multiplication.

*   **The Code (`src/layers.py`):
    The `forward` method orchestrates the `im2col` transformation:
    ```python
    def forward(self, input):
        # ...
        # Reshape weights for matrix multiplication
        weights_reshaped = self.weights.reshape(-1, self.num_filters)

        # Convert input to im2col matrix
        self.im2col_matrix = self._im2col(self.input_padded)

        # Perform matrix multiplication (the convolution!)
        output_im2col = np.dot(self.im2col_matrix, weights_reshaped)
        # ... reshape output and add bias
        return self.output
    ```
    The backward pass uses the same principles in reverse, leveraging the stored `im2col_matrix` and the `_col2im` helper function to calculate gradients efficiently.

---

## `MaxPooling` Layer

*   **What it is:** A down-sampling layer. It slides a window over its input and, for each window, outputs only the single maximum value. This reduces the size of the data, speeds up computation, and makes the network more robust to the exact location of features.

*   **The Math:**
    *   **Forward Pass:** For each window, $ \text{output} = \max(\text{window}) $
    *   **Backward Pass:** The gradient is passed back *only* to the neuron that had the maximum value in the forward pass. All other neurons in that window receive a gradient of zero.

*   **The Code (`src/layers.py`):
    The forward pass finds and stores the index of the max value:
    ```python
    def forward(self, input):
        # ... loops through windows ...
        patch = input[b, h_start:h_end, w_start:w_end, c]
        max_value = np.max(patch)
        max_idx_flat = np.argmax(patch)
        # ... store the index of the max value ...
        self.output[b, h, w, c] = max_value
        self.max_indices[b, h, w, c] = (max_h_rel, max_w_rel)
    ```
    The backward pass uses these stored indices to route the gradients:
    ```python
    def backward(self, output_gradient, learning_rate):
        # ... loops through gradients ...
        # Get the stored relative index of the max value
        max_h_rel, max_w_rel = self.max_indices[b, h, w, c]

        # Propagate gradient only to the max element
        input_gradient[b, h_start + max_h_rel, w_start + max_w_rel, c] += output_gradient[b, h, w, c]
    ```

---

## `Flatten` Layer

*   **What it is:** A simple utility layer that reshapes a multi-dimensional input (like the output of a convolutional layer) into a 1D vector. This is necessary to connect the feature-detecting convolutional part of the network to the classification-performing dense part.

*   **The Code (`src/layers.py`):
    ```python
    def forward(self, input):
        self.input_shape = input.shape
        self.output = input.reshape(input.shape[0], -1)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
    ```
    The forward pass flattens the data, and the backward pass simply reshapes the incoming gradient back to the original input shape.
