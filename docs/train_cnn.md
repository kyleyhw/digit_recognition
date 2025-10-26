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

### Design Rationale for LeNet-5 Architecture

The LeNet-5 inspired architecture used in this project is particularly well-suited for digit recognition tasks like MNIST due to several key design principles:

*   **Spatial Feature Extraction:** The initial convolutional layers are designed to learn local features such as edges, corners, and textures. By stacking these layers, the network progressively learns more complex and abstract representations of the input image.
*   **Dimensionality Reduction and Translation Invariance:** MaxPooling layers reduce the spatial dimensions of the feature maps, which helps in making the model more robust to small shifts or distortions in the input image. This downsampling also reduces the computational cost in subsequent layers.
*   **Feature Learning to Classification Transition:** The Flatten layer converts the 3D feature maps into a 1D vector, preparing the data for the fully connected (Dense) layers. These Dense layers then act as a classifier, mapping the learned features to the final digit predictions.
*   **Non-linearity for Complex Patterns:** ReLU activation functions introduce non-linearity into the network, allowing it to learn complex, non-linear boundaries in the feature space that are necessary for distinguishing between different handwritten digits.
*   **Probabilistic Output:** The final Dense layer followed by a Softmax activation produces a probability distribution over the 10 possible digit classes, making the output easily interpretable.

#### Alternatives to Considered Architectures

While the LeNet-5 style is effective, several other architectures exist, each with its own advantages and disadvantages:

1.  **Purely Fully Connected Networks (FCNs):**
    *   **Description:** Each pixel would be an input to the first layer, and layers would consist solely of Dense layers.
    *   **Pros:** Conceptually simpler for basic non-image tasks.
    *   **Cons:**
        *   **Loss of Spatial Information:** FCNs treat pixels as independent features, losing critical spatial relationships (e.g., a handwritten '7' has a distinct pattern of connected pixels).
        *   **Massive Parameter Count:** For image data, an FCN requires a weight for every connection from every input pixel to every neuron in the first hidden layer. For a 28x28 image, this leads to an explosion of parameters, making the network computationally expensive, harder to train, and highly prone to overfitting.
        *   **No Translational Invariance:** Changes in an object's position require the network to learn new features for each position.

2.  **Deeper and More Complex CNN Architectures (e.g., VGG, ResNet, Inception):**
    *   **Description:** These are state-of-the-art CNNs with many more layers, often incorporating advanced concepts like residual connections (ResNet) or multi-scale processing (Inception).
    *   **Pros:** Achieve significantly higher accuracy on complex datasets (e.g., ImageNet) by learning richer, more abstract features. They can capture intricate patterns and generalize better on diverse real-world images.
    *   **Cons:**
        *   **Computational Expense:** Require immense computational resources (GPUs, TPUs) and long training times.
        *   **Data Hunger:** Need vast amounts of labeled data for effective training from scratch.
        *   **Increased Complexity:** More challenging to implement from scratch and understand the precise role of each component.
        *   **Overkill for Simple Tasks:** For a simple task like MNIST, their complexity is often unnecessary.

### Adapting CNNs to Other Applications

The foundational principles of CNNs (hierarchical feature learning, parameter sharing, local receptive fields) make them highly versatile for various computer vision tasks:

1.  **Image Classification:** (As in this project) Assigning a label to an entire image (e.g., identifying animals, scenes, objects). By adjusting the output layer to match the number of classes and training on relevant datasets, a CNN can categorize any type of image.

2.  **Object Detection:** Identifying and locating specific objects within an image, typically by drawing bounding boxes around them and assigning a class label to each box. This involves extensions to CNNs that predict both bounding box coordinates and class probabilities (e.g., R-CNN, YOLO, SSD). The convolutional backbone is often used as a feature extractor.

3.  **Image Segmentation:** Performing pixel-level classification, where each pixel in an image is assigned a class label (e.g., "road," "car," "pedestrian," "background"). Architectures like U-Net or Fully Convolutional Networks (FCNs) are commonly used, building upon CNN features to reconstruct a dense classification map. Useful in medical imaging (tumor segmentation) and autonomous driving.

4.  **Generative Models:** Creating new images that are indistinguishable from real images, or generating images based on certain conditions (e.g., text descriptions). Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) utilize CNN components to synthesize realistic imagery.

5.  **Transfer Learning:** Reusing a pre-trained CNN (trained on a very large, general dataset like ImageNet) as a starting point for a new, often more specific, vision task. The early layers, which learn generic features (edges, textures), can be kept frozen, and only the later layers (or new layers added on top) are fine-tuned on the new, smaller dataset. This significantly reduces the amount of data and computation needed for training.

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

## Training Hyperparameters and Initialization

### Number of Epochs (10)

An **epoch** represents one full pass over the entire training dataset. The model iteratively learns from the data, adjusts its parameters, and then repeats the process for a specified number of epochs.

*   **Chosen Value:** In `train_mnist.py`, the model is currently configured to train for `10` epochs.
*   **Rationale:** This number was chosen as a practical starting point for several reasons:
    *   **Educational Purpose:** For a from-scratch implementation running on a CPU, `10` epochs allow the network to demonstrate clear learning convergence without requiring an excessively long training time, especially when using the full MNIST dataset. Longer training times could hinder iterative development and experimentation.
    *   **Subset Training:** When training on the reduced dataset (e.g., 1000 samples), 10 epochs are often sufficient to observe significant improvements in loss and accuracy.
    *   **Avoid Overfitting (Initial Stage):** While more epochs might lead to higher accuracy on the training set, a higher number could also lead to overfitting, especially with small datasets or simple models. `10` epochs provide a reasonable balance to learn general features without memorizing the training data.

### Learning Rate (0.01)

The **learning rate** is a hyperparameter that determines the step size at each iteration while moving toward a minimum of the loss function. It dictates how quickly or slowly the model adjusts its weights with respect to the gradient of the loss function.

*   **Chosen Value:** The learning rate is set to `0.01` in `train_mnist.py`.
*   **Rationale:** `0.01` is a commonly used default starting value in many deep learning tasks and optimization algorithms. Its selection is based on practical heuristics:
    *   **Balance:** It's often a good compromise between making rapid progress (higher learning rate) and ensuring stability (lower learning rate).
    *   **Too High:** A learning rate that is too high can cause the optimization process to overshoot the optimal weights, oscillate, or even diverge, leading to an unstable or non-convergent training.
    *   **Too Low:** Conversely, a learning rate that is too low will make the training process very slow, requiring many more epochs to converge, potentially getting stuck in local minima.
*   **Dynamic Adjustment:** In more advanced scenarios, the learning rate can be dynamically adjusted during training (e.g., learning rate schedules, adaptive optimizers like Adam or RMSprop) to achieve better convergence and performance. For this foundational implementation, a fixed rate provides simplicity.

### Parameter Initialization

Parameter initialization refers to the process of setting the initial values of the weights ($\mathbf{W}$) and biases ($\mathbf{b}$) for all layers in the neural network before training begins.

*   **Technique Used:** In our `Convolutional` and `Dense` layers, weights are typically initialized from a **random normal distribution** with a small standard deviation, and biases are initialized to **zeros**.
*   **Rationale:**
    *   **Breaking Symmetry (Weights):** If all weights were initialized to the same value (e.g., all zeros), every neuron in a layer would learn the exact same features during training. This symmetry would prevent the network from learning diverse and meaningful representations. Random initialization ensures that each neuron starts in a unique state, allowing them to differentiate and specialize.
    *   **Preventing Vanishing/Exploding Gradients (Weights):** Initializing weights with appropriate scaling helps to prevent the gradients from becoming extremely small (vanishing) or extremely large (exploding) as they propagate backward through the network. While a simple random normal distribution is used here for simplicity, more sophisticated methods like Xavier/Glorot or He initialization (which scale random initializations based on the number of input/output neurons) are often used in practice to maintain healthy gradient flow, especially in deep networks.
    *   **Zero Biases:** Initializing biases to zero is a common practice and generally doesn't suffer from the symmetry problem that zero weights would. Biases serve to shift the activation function, and starting from zero allows the weights to primarily drive the initial learning process.

### General Guidelines for Choosing Hyperparameters

Selecting optimal hyperparameters like the number of epochs, batch size, and learning rate is often more an art than a science, involving a mix of theoretical understanding, empirical experimentation, and domain knowledge. Here are some general guidelines:

#### Choosing the Number of Epochs

*   **Definition:** One epoch is a complete pass through the entire training dataset. The model completes one full update cycle (forward pass, loss calculation, backward pass, parameter update) for every training example.
*   **Too Few Epochs (Underfitting):** If you train for too few epochs, the model will not have had enough time to learn the underlying patterns in the data effectively. It will perform poorly on both the training and test sets.
*   **Too Many Epochs (Overfitting):** Training for too many epochs can lead to overfitting, where the model learns the training data too well, memorizing noise and specific examples rather than general patterns. This results in good performance on the training set but poor generalization to new, unseen data.
*   **Monitoring and Early Stopping:** The most effective way to choose the number of epochs is to monitor the model's performance on a separate **validation set** during training. You typically stop training when the performance on the validation set starts to degrade or plateau, even if the training loss continues to decrease. This technique is called **early stopping**.
*   **Consider Data Size and Complexity:** Larger datasets or more complex models generally require more epochs to converge. Simpler models or smaller datasets might overfit quickly.

#### Choosing the Batch Size

*   **Definition:** The batch size is the number of training examples utilized in one iteration. The model's parameters are updated after processing each batch.
*   **Small Batch Size (e.g., 1-32):**
    *   **Pros:** Leads to a noisy but potentially better estimate of the gradient, which can help escape shallow local minima. Requires less memory. Acts as a regularizer, preventing overfitting. Good for online learning or very large datasets.
    *   **Cons:** Training can be slower (more weight updates per epoch). Gradient estimates are noisy, leading to more oscillations in validation loss.
*   **Large Batch Size (e.g., 64-256+):**
    *   **Pros:** Provides a more stable and accurate estimate of the gradient. Leads to faster training computation per epoch (more efficient use of hardware). Smoother convergence. Less susceptible to noisy gradients.
    *   **Cons:** Can get stuck in sharp local minima. Requires more memory. Can sometimes lead to poorer generalization by converging to flatter minima that are further from the true optimum. More prone to overfitting if not regularized appropriately.
*   **Common Practice:** Batch sizes that are powers of 2 (e.g., 16, 32, 64, 128) are often empirically found to be computationally efficient due to how hardware (especially GPUs) processes data.

#### Choosing the Learning Rate

*   **Definition:** The learning rate ($\eta$) is arguably the most important hyperparameter. It scales the magnitude of the weight updates during gradient descent. Effectively, it controls how aggressively the model learns.
*   **Too High Learning Rate:**
    *   **Effect:** The model's parameters might overshoot the optimal point (the minimum of the loss function) or even diverge, causing the loss to increase or oscillate wildly.
    *   **Symptoms:** Rapidly increasing or `NaN` loss values.
*   **Too Low Learning Rate:**
    *   **Effect:** Training becomes very slow, and the model might get stuck in a suboptimal local minimum. It might take an extremely long time to converge.
    *   **Symptoms:** Loss decreases very slowly, or plateaus at a high value.
*   **Finding a Good Learning Rate:**
    *   **Learning Rate Finder:** Techniques like Leslie Smith's learning rate finder can be used to empirically find a good range for the learning rate by briefly training the model with exponentially increasing learning rates.
    *   **Log-Scale Search:** Systematically trying values on a log scale (e.g., 0.1, 0.01, 0.001) is a common starting point.
    *   **Learning Rate Schedules:** Changing the learning rate during training (e.g., reducing it after a certain number of epochs, or when validation loss plateaus) can improve convergence and final performance.
    *   **Adaptive Optimizers:** Optimizers like Adam, RMSprop, or Adagrad automatically adjust the learning rate for each parameter during training, reducing the need for manual tuning of a global learning rate.

Effectively tuning these hyperparameters is crucial for training stable, efficient, and well-performing neural networks.

## Checkpointing (Pausing and Resuming Training)

For long training runs, it's crucial to be able to save the model's progress and resume training later. Our `Network` class supports this through checkpointing:

*   **Saving Checkpoints:** During training, after each epoch, the model's current weights and biases are saved to a `.npz` file in the top-level `checkpoints/` directory. Each checkpoint file is named using a prefix (e.g., `model_full_epoch_1.npz` or `model_subset_plot_epoch_1.npz`) to distinguish between different training runs.
*   **Resuming Training:** If `resume_from_checkpoint` is set to `True` in the `train` method, the network will automatically look for the latest checkpoint file with the specified `checkpoint_prefix` in the `checkpoints/` directory. If found, it loads the parameters from that checkpoint and continues training from the next epoch.

This allows you to interrupt training (e.g., if your computer shuts down or you need to use it for something else) and pick up exactly where you left off, saving valuable training time.

## GPU Usage (CPU-Only Implementation)

It's important to note that this "CNN from Scratch" implementation, built using pure NumPy, runs exclusively on the **CPU (Central Processing Unit)**.

*   **Why CPU-Only?** The primary goal of this project is to understand the fundamental mathematics and algorithms of neural networks by implementing them from scratch. NumPy provides the necessary array manipulation capabilities for this educational purpose.
*   **GPU Acceleration:** Modern deep learning frameworks (like PyTorch, TensorFlow, JAX) achieve significant speedups by leveraging **GPUs (Graphics Processing Units)**. GPUs are highly parallel processors optimized for the matrix multiplications and other numerical computations common in neural networks.
*   **How GPUs Work (Briefly):** To utilize a GPU, the numerical operations would need to be offloaded to a GPU-accelerated library (e.g., NVIDIA's CuPy, which provides a NumPy-like interface for GPUs) or a full deep learning framework. This involves managing memory on the GPU and using GPU-specific kernels for computations.
*   **Implication for this Project:** While our implementation is functional and educational, it will be considerably slower than a GPU-accelerated version, especially for larger datasets or more complex models. This is an inherent trade-off when building from first principles with CPU-bound libraries.
