# Network Class Documentation

The `Network` class serves as the central orchestrator for building and training sequential neural network models. It provides a simple interface to add layers, perform forward and backward passes, and manage the training process.

## Class Definition

```python
class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function):
        self.loss_function = loss_function

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, X_train, y_train, epochs, learning_rate, batch_size=32):
        if self.loss_function is None:
            raise ValueError("Loss function not compiled. Call network.compile(loss_function) first.")

        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            total_loss = 0
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                loss = self.loss_function.loss(y_batch, output)
                total_loss += loss

                output_gradient = self.loss_function.prime(y_batch, output)
                self.backward(output_gradient, learning_rate)
            
            avg_loss = total_loss / (num_samples / batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, input_data):
        return self.forward(input_data)
```

## Methods

### `__init__(self)`

*   **Purpose:** Initializes a new `Network` instance.
*   **Description:** Creates an empty list `self.layers` to store the network layers and initializes `self.loss_function` to `None`.

### `add(self, layer)`

*   **Purpose:** Adds a layer to the neural network.
*   **Parameters:**
    *   `layer`: An instance of a class inheriting from `Layer` (e.g., `Dense`, `ReLU`, `Convolutional`).
*   **Description:** Appends the provided `layer` object to the `self.layers` list. Layers are processed in the order they are added.

### `compile(self, loss_function)`

*   **Purpose:** Configures the model for training.
*   **Parameters:**
    *   `loss_function`: An instance of a class inheriting from `Loss` (e.g., `MeanSquaredError`).
*   **Description:** Assigns the specified loss function to the network. This method must be called before training.

### `forward(self, input_data)`

*   **Purpose:** Performs the forward pass through all layers in the network.
*   **Parameters:**
    *   `input_data`: The input data for the network (e.g., a batch of images).
*   **Returns:** The output of the last layer after processing the `input_data` through the entire network.
*   **Description:** Iterates through each layer in `self.layers` sequentially, passing the output of the current layer as the input to the next layer.

### `backward(self, output_gradient, learning_rate)`

*   **Purpose:** Performs the backward pass (backpropagation) through all layers in reverse order.
*   **Parameters:**
    *   `output_gradient`: The gradient of the loss function with respect to the output of the last layer. This is typically obtained from the derivative of the chosen loss function.
    *   `learning_rate`: The learning rate used to update the trainable parameters (weights and biases) of the layers during backpropagation.
*   **Description:** Iterates through the layers in `self.layers` in reverse order. For each layer, it calls its `backward` method, passing the `output_gradient` from the subsequent layer and the `learning_rate`. The `backward` method of each layer computes the gradients with respect to its own parameters and its input, and then returns the gradient with respect to its input, which becomes the `output_gradient` for the preceding layer.

### `train(self, X_train, y_train, epochs, learning_rate, batch_size=32)`

*   **Purpose:** Orchestrates the training loop of the neural network using mini-batch stochastic gradient descent (SGD).
*   **Parameters:**
    *   `X_train`: Training input data.
    *   `y_train`: Training target labels.
    *   `epochs`: The number of times to iterate over the entire training dataset.
    *   `learning_rate`: The learning rate for parameter updates.
    *   `batch_size`: The number of samples per gradient update.
*   **Description:** This method iterates through the specified number of epochs. In each epoch, it shuffles the training data and processes it in mini-batches. For each batch, it performs a forward pass to get predictions, calculates the loss, and then performs a backward pass to update the network's parameters. The average loss for each epoch is printed to the console.

### `predict(self, input_data)`

*   **Purpose:** Makes predictions using the trained network.
*   **Parameters:**
    *   `input_data`: The input data for which to make predictions.
*   **Returns:** The network's output (predictions) for the given input data.
*   **Description:** This method performs a forward pass through the network without any gradient calculations or parameter updates.
