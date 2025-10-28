# Network Class Documentation

The `Network` class is the heart of our neural network library. It acts as a container for a sequence of layers, and provides the core logic to train the model on data and make predictions.

(See [Fundamental Concepts](./Concepts.md) for an introduction to the core ideas of a neural network.)

## Core Concepts

*   **Sequential Model:** This `Network` class represents a sequential model, which is a linear stack of layers. You simply `.add()` layers one after another to build the model architecture [3].
*   **Training Loop:** The process of training a neural network involves repeatedly showing it data, allowing it to make predictions, comparing those predictions to the true targets (calculating the **loss**), and then adjusting the internal parameters (weights and biases) to reduce that loss. This adjustment process is called **backpropagation** and is driven by an **optimizer** (in our case, Stochastic Gradient Descent) [3].
*   **Epoch:** One complete pass through the entire training dataset [3].
*   **Batch:** Instead of processing the entire dataset at once, the data is broken into smaller chunks called batches. The network's parameters are updated after each batch. This is known as **mini-batch gradient descent** [3].

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
        # ... (Implementation details in file)

    def predict(self, input_data):
        return self.forward(input_data)
```

## Methods

### `__init__(self)`

*   **Purpose:** Initializes a new, empty `Network` instance.
*   **Description:** Creates an empty list `self.layers` which will store the neural network layers in sequential order, and initializes `self.loss_function` to `None`.

### `add(self, layer)`

*   **Purpose:** Adds a layer to the network.
*   **Parameters:**
    *   `layer`: An instance of a class inheriting from `Layer` (e.g., `Dense`, `ReLU`, `Convolutional`).
*   **Description:** Appends the provided `layer` object to the `self.layers` list. The order in which layers are added defines the architecture of the network.

### `compile(self, loss_function)`

*   **Purpose:** Configures the model for training.
*   **Parameters:**
    *   `loss_function`: An instance of a class inheriting from `Loss` (e.g., `MeanSquaredError`).
*   **Description:** Assigns the specified loss function to the network. This method must be called before training can begin.

### `forward(self, input_data)`

*   **Purpose:** Performs the forward pass, propagating input data through all layers in the network.
*   **Parameters:**
    *   `input_data`: The input data for the network (e.g., a batch of images).
*   **Returns:** The output of the final layer after processing the `input_data` through the entire network.
*   **Description:** This method takes an input and passes it through each layer sequentially, with the output of one layer becoming the input for the next.

### `backward(self, output_gradient, learning_rate)`

*   **Purpose:** Performs the backward pass (backpropagation) through all layers in reverse order.
*   **Parameters:**
    *   `output_gradient`: The gradient of the loss function with respect to the output of the last layer. This is the starting point for backpropagation.
    *   `learning_rate`: A hyperparameter that controls how much the weights and biases are adjusted during training. A smaller learning rate leads to slower but potentially more stable learning.
*   **Description:** This method propagates the error gradient backward through the network. It starts with the gradient from the loss function and passes it to the last layer. Each layer then calculates the gradient with respect to its own parameters (and updates them), and passes the gradient with respect to its input to the previous layer in the sequence.

### `train(self, X_train, y_train, epochs, learning_rate, batch_size=32)`

*   **Purpose:** Orchestrates the full training loop of the neural network using mini-batch stochastic gradient descent (SGD).
*   **Parameters:**
    *   `X_train`: The set of training input data.
    *   `y_train`: The set of corresponding true target labels.
    *   `epochs`: The number of times the entire training dataset is passed through the network.
    *   `learning_rate`: The step size for the optimizer.
    *   `batch_size`: The number of training samples to process before updating the model's parameters.
*   **Description:** This is the main engine of learning. It iterates through the specified number of epochs. In each epoch, it shuffles the training data and processes it in mini-batches. For each batch, it performs a forward pass to get predictions, calculates the loss and its gradient, and then performs a backward pass to update all the network's trainable parameters.

### `predict(self, input_data)`

*   **Purpose:** Makes predictions on new, unseen data using the trained network.
*   **Parameters:**
    *   `input_data`: The input data for which to make predictions.
*   **Returns:** The network's final output (e.g., class probabilities from a Softmax layer).
*   **Description:** This method simply performs a forward pass through the network. It does not perform any learning or parameter updates.

### `save_model(self, filepath)`

*   **Purpose:** Saves the weights and biases of all layers with parameters to a `.npz` file.
*   **Parameters:**
    *   `filepath`: The path to the file where the model parameters will be saved.
*   **Description:** Iterates through all layers in the network and saves the `weights` and `bias` arrays from each layer that has them (e.g., `Dense`, `Convolutional`) into a single compressed NumPy file.

### `load_model(self, filepath)`

*   **Purpose:** Loads weights and biases from a `.npz` file into the network layers.
*   **Parameters:**
    *   `filepath`: The path to the file from which to load the model parameters.
*   **Description:** Loads the parameters from the specified file and assigns them to the corresponding layers in the network. The network architecture defined in the code must be identical to the one used when the model was saved.

## References

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
[2] LeCun, Y., & Cortes, C. (1998). The MNIST database of handwritten digits.
[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
[4] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, 249-256.
[5] Chellapilla, K., Puri, S., & Simard, P. (2006). High Performance Convolutional Neural Networks for Document Processing. *Tenth International Workshop on Frontiers in Handwriting Recognition*.
