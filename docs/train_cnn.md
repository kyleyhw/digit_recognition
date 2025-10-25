# Training a CNN on MNIST

This document explains the `train_mnist.py` script, which serves as the main entry point for building, training, and evaluating our from-scratch Convolutional Neural Network (CNN) on the MNIST dataset.

(For a detailed breakdown of each layer, see [The Layers Explained](./Layers_Explained.md). For an introduction to the core concepts, see [Fundamental Concepts](./Concepts.md).)

## The `train_mnist.py` Script

*   **Purpose:** To bring together all the implemented components (layers, loss functions, network) to create and train a functional CNN for handwritten digit recognition.
*   **Workflow:**
    1.  **Data Loading:** Loads the preprocessed MNIST data using the `preprocess_mnist_data` function. For faster development and testing, the script is initially configured to use a smaller subset of the data (1000 training samples, 200 test samples).
    2.  **Model Definition:** A CNN model is constructed by adding a sequence of layers to a `Network` object.
    3.  **Compilation:** The network is compiled with the `CategoricalCrossEntropy` loss function, which is the standard for multi-class classification.
    4.  **Training:** The `network.train()` method is called to train the model on the training data for a specified number of epochs and with a given learning rate.
    5.  **Evaluation:** After training, the script iterates through the test set, makes predictions for each image, and compares the predicted class with the true class to calculate the final accuracy.

## CNN Architecture (LeNet-5 Style)

The architecture is inspired by the classic LeNet-5 model, which is highly effective for digit recognition.

1.  **Input:** `(28, 28, 1)` - Grayscale images of handwritten digits.

2.  **Layer 1: Convolutional -> ReLU -> MaxPooling**
    *   `Convolutional(input_shape=(28, 28, 1), num_filters=6, kernel_size=(5, 5), padding='same')`: Applies 6 different 5x5 filters to the input image. `'same'` padding is used to ensure the output feature maps have the same spatial dimensions (28x28) as the input.
    *   `ReLU()`: Introduces non-linearity.
    *   `MaxPooling(pool_size=(2, 2), stride=(2, 2))`: Downsamples the 28x28 feature maps to 14x14, reducing dimensionality and providing translational invariance.
    *   *Output Shape:* `(14, 14, 6)`

3.  **Layer 2: Convolutional -> ReLU -> MaxPooling**
    *   `Convolutional(input_shape=(14, 14, 6), num_filters=16, kernel_size=(5, 5), padding='valid')`: Applies 16 different 5x5 filters to the feature maps from the previous layer. `'valid'` padding means no padding is applied.
    *   `ReLU()`: Introduces non-linearity.
    *   `MaxPooling(pool_size=(2, 2), stride=(2, 2))`: Downsamples the 10x10 feature maps to 5x5.
    *   *Output Shape:* `(5, 5, 16)`

4.  **Layer 3: Flatten**
    *   `Flatten()`: Converts the 3D feature maps `(5, 5, 16)` into a 1D vector.
    *   *Output Shape:* `(400,)` (since 5 * 5 * 16 = 400)

5.  **Layer 4: Dense -> ReLU**
    *   `Dense(400, 120)`: A fully connected layer that maps the 400 features to 120 features.
    *   `ReLU()`: Introduces non-linearity.
    *   *Output Shape:* `(120,)`

6.  **Layer 5: Dense -> ReLU**
    *   `Dense(120, 84)`: Another fully connected layer, reducing the features from 120 to 84.
    *   `ReLU()`: Introduces non-linearity.
    *   *Output Shape:* `(84,)`

7.  **Layer 6: Output Layer (Dense -> Softmax)**
    *   `Dense(84, 10)`: The final fully connected layer, which produces 10 output values (one for each digit class).
    *   `Softmax()`: Converts the 10 raw output values into a probability distribution, where each value represents the predicted probability for a digit from 0 to 9.
    *   *Output Shape:* `(10,)`

## How to Run

To run the entire training and evaluation process, execute the following command from your project's root directory:

```bash
python train_mnist.py
```

The script will print the training loss for each epoch and, upon completion, will display the final test accuracy.

## Key Concepts in this Script

*   **Convolutional Layer:** The core building block of a CNN. It uses a set of learnable filters (or kernels) to detect features like edges, corners, and textures in an image.
*   **ReLU (Rectified Linear Unit):** A simple yet powerful activation function that introduces non-linearity by outputting the input directly if it is positive, and zero otherwise. This helps the network learn more complex patterns.
*   **MaxPooling Layer:** A down-sampling technique that reduces the spatial dimensions of the feature maps. It works by taking the maximum value over a window of the input, which helps to make the learned features more robust to small translations in the input image.
*   **Flatten Layer:** A utility layer that reshapes a multi-dimensional tensor into a 1D vector. This is a necessary step to transition from the convolutional/pooling layers to the fully connected dense layers.
*   **Dense Layer:** A standard fully connected neural network layer where each input neuron is connected to each output neuron. These layers are typically used in the final stages of a CNN to perform classification based on the features extracted by the convolutional layers.
*   **Softmax Layer:** An activation function used in the output layer for multi-class classification. It converts the raw output scores (logits) from the final dense layer into a probability distribution over the classes, where all probabilities sum to 1.
*   **Categorical Cross-Entropy Loss:** The loss function used for multi-class classification. It measures the difference between the predicted probability distribution (from the Softmax layer) and the true distribution (the one-hot encoded label).
