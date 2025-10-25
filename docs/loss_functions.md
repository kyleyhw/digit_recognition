# Loss Functions Documentation

Loss functions are a crucial component of neural networks, quantifying the discrepancy between the network's predictions and the true target values. During training, the goal of the optimization algorithm is to minimize this loss.

## Base Class: `Loss`

The `Loss` class serves as an abstract base for all loss functions. It defines the common interface that all specific loss functions must implement.

### Class Definition

```python
class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def prime(self, y_true, y_pred):
        raise NotImplementedError
```

### Methods

#### `loss(self, y_true, y_pred)`

*   **Purpose:** Calculates the value of the loss function.
*   **Parameters:**
    *   `y_true`: The true target values.
    *   `y_pred`: The predicted values from the neural network.
*   **Returns:** A scalar value representing the calculated loss.
*   **Description:** This method must be implemented by subclasses to compute the specific loss value based on the true and predicted outputs.

#### `prime(self, y_true, y_pred)`

*   **Purpose:** Calculates the derivative of the loss function with respect to the predicted values (`y_pred`).
*   **Parameters:**
    *   `y_true`: The true target values.
    *   `y_pred`: The predicted values from the neural network.
*   **Returns:** An array of the same shape as `y_pred`, representing the gradient of the loss with respect to each predicted value.
*   **Description:** This method must be implemented by subclasses to compute the gradient of the loss. This derivative is the `output_gradient` that is passed to the `backward` method of the last layer in the neural network during backpropagation.

---

## Specific Loss Functions

### `MeanSquaredError`

The `MeanSquaredError` (MSE) loss function is a common choice for regression problems. It quantifies the average squared difference between the predicted values and the true values.

#### Class Definition

```python
class MeanSquaredError(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_pred - y_true, 2))

    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
```

#### Mathematical Explanation

*   **Forward Pass (Loss Calculation):**
    The formula for MSE is:
    $$ L = \frac{1}{N} \sum_{i=1}^{N} (y_{pred,i} - y_{true,i})^2 $$
    Where:
    *   $N$ is the number of samples.
    *   $y_{pred,i}$ is the predicted value for the $i$-th sample.
    *   $y_{true,i}$ is the true value for the $i$-th sample.
    The errors are squared to ensure that positive and negative errors contribute equally to the loss, and to penalize larger errors more heavily. The mean is taken over all samples.

*   **Backward Pass (Derivative of Loss):**
    The derivative of the MSE loss function with respect to the predicted values (`y_pred`) is:
    $$ \frac{\partial L}{\partial y_{pred}} = \frac{2}{N} (y_{pred} - y_{true}) $$
    This derivative is used as the `output_gradient` for the last layer during backpropagation.

---


### `CategoricalCrossEntropy`

The `CategoricalCrossEntropy` loss function is the standard choice for multi-class classification problems, especially when the output layer uses a `Softmax` activation function. It measures the difference between two probability distributions: the true labels (typically one-hot encoded) and the predicted probabilities.

#### Class Definition

```python
class CategoricalCrossEntropy(Loss):
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
```

#### Mathematical Explanation

*   **Forward Pass (Loss Calculation):**
    The formula for Categorical Cross-Entropy is:
    $$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{true,ic} \log(y_{pred,ic}) $$
    Where:
    *   $N$ is the number of samples.
    *   $C$ is the number of classes.
    *   $y_{true,ic}$ is 1 if sample $i$ belongs to class $c$, and 0 otherwise (one-hot encoded).
    *   $y_{pred,ic}$ is the predicted probability that sample $i$ belongs to class $c$.
    The logarithm heavily penalizes incorrect predictions, especially when the predicted probability for the true class is very low. The negative sign makes the loss positive.

*   **Backward Pass (Derivative of Loss):**
    The derivative of the Categorical Cross-Entropy loss with respect to the predicted probabilities (`y_pred`) is:
    $$ \frac{\partial L}{\partial y_{pred}} = -\frac{1}{N} \frac{y_{true}}{y_{pred}} $$
    **Important Note:** When Categorical Cross-Entropy Loss is used immediately after a `Softmax` activation function, the combined derivative of the `Softmax` layer and the `CategoricalCrossEntropy` loss simplifies significantly to:
    $$ \frac{\partial L}{\partial \text{input to Softmax}} = y_{pred} - y_{true} $$
    This simplification is a key reason why these two are almost always used together. The `prime` method here calculates the derivative of the loss *with respect to the output of the Softmax layer*.

```