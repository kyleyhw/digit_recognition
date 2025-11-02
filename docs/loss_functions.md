# Loss Functions Documentation

Loss functions are a crucial component of neural networks, quantifying the discrepancy between the network's predictions and the true target values. During training, the goal of the optimization algorithm is to minimize this loss [3].

(See [Fundamental Concepts](./concepts.md) for an introduction to Loss Functions, Training, and Optimization.)

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
    *   `y_true`: The ground-truth target values (e.g., the actual digit labels).
    *   `y_pred`: The predicted values from the neural network.
*   **Returns:** A scalar value representing the calculated loss for a batch of data.

#### `prime(self, y_true, y_pred)`

*   **Purpose:** Calculates the derivative (gradient) of the loss function with respect to the predicted values (`y_pred`).
*   **Returns:** An array of the same shape as `y_pred`, representing the gradient of the loss.
*   **Description:** This is the starting point for the backpropagation algorithm. It tells the final layer of the network how much it needs to change its output to reduce the loss.

---

## Specific Loss Functions

### `MeanSquaredError` (MSE)

The `MeanSquaredError` (MSE) loss function is primarily used for **regression problems**, where the goal is to predict a continuous value (e.g., the price of a house) [3]. It quantifies the average squared difference between the predicted and true values.

#### Mathematical Explanation

*   **Forward Pass (Loss Calculation):**
    The formula for MSE is:
    $ L = \frac{1}{N} \sum_{i=1}^{N} (y_{pred,i} - y_{true,i})^2 $
    Where $N$ is the number of samples. The errors are squared to ensure that positive and negative errors contribute equally and to penalize larger errors more heavily [3].

*   **Backward Pass (Derivative of Loss):**
    The derivative of the MSE loss with respect to the predicted values is:
    $ \frac{\partial L}{\partial y_{pred}} = \frac{2}{N} (y_{pred} - y_{true}) $

---

### `CategoricalCrossEntropy`

The `CategoricalCrossEntropy` loss function is the standard choice for **multi-class classification problems**, such as our MNIST digit recognition task [3]. It is designed to work with a `Softmax` activation function on the output layer [3].

#### Core Concept: Cross-Entropy

Cross-entropy measures the difference between two probability distributions [3]. In our case, these are:
1.  The **true distribution**: The one-hot encoded label (e.g., for the digit '2', the distribution is `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`).
2.  The **predicted distribution**: The output of the `Softmax` layer (e.g., `[0.05, 0.1, 0.7, 0.05, ...]`)

The loss is low when the predicted distribution is very similar to the true distribution (i.e., a high probability for the correct class).

#### Mathematical Explanation

*   **Forward Pass (Loss Calculation):**
    The formula for Categorical Cross-Entropy is:
    $ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{true,ic} \log(y_{pred,ic}) $
    Where $N$ is the number of samples and $C$ is the number of classes. Since `y_true` is one-hot encoded, this simplifies to just taking the negative log of the predicted probability for the single correct class [3].

*   **Backward Pass (Derivative of Loss):**
    The derivative of the Categorical Cross-Entropy loss with respect to the predicted probabilities (`y_pred`) is:
    $ \frac{\partial L}{\partial y_{pred}} = -\frac{1}{N} \frac{y_{true}}{y_{pred}} $
    **Important Note:** When this loss is used immediately after a `Softmax` activation function, the combined derivative (of `Softmax` and `CategoricalCrossEntropy`) simplifies to the very elegant and stable form:
    $ \frac{\partial L}{\partial \text{input to Softmax}} = y_{pred} - y_{true} $
    This simplification is a key reason why these two are almost always used together in classification tasks [3].

---

## References

[1]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[2]: http://yann.lecun.com/exdb/mnist/
[3]: https://www.deeplearningbook.org/
[4]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
[5]: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/04/chellapilla-simard-puri-IWFHR-2006.pdf

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
[2] LeCun, Y., & Cortes, C. (1998). The MNIST database of handwritten digits.
[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
[4] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, 249-256.
[5] Chellapilla, K., Puri, S., & Simard, P. (2006). High Performance Convolutional Neural Networks for Document Processing. *Tenth International Workshop on Frontiers in Handwriting Recognition*.