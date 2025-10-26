# Fundamental Concepts of Neural Networks

This document introduces the core ideas of neural networks for readers with a mathematical background but no prior experience in machine learning.

## What is a Neural Network?

A neural network is a computational model inspired by the structure of the human brain. At its core, it is a mathematical function that learns to map inputs to outputs. For our project, the input is an image of a handwritten digit, and the output is the network's prediction of what that digit is (0-9).

The network is composed of interconnected "neurons" organized into **layers**.

## The Neuron

A neuron is the basic computational unit. It receives one or more inputs, performs a simple calculation, and produces an output. The calculation involves two steps:

1.  **Linear Combination:** The neuron takes its inputs ($x_1, x_2, ...$), multiplies each by a corresponding **weight** ($w_1, w_2, ...$), sums them up, and adds a **bias** ($b$).
    $$ z = (w_1 x_1 + w_2 x_2 + \dots) + b = \mathbf{w} \cdot \mathbf{x} + b $$
    *   **Weights (w):** These are learnable parameters that determine the strength of the connection between neurons. A higher weight means the input has more influence.
    *   **Bias (b):** This is another learnable parameter that allows the neuron to shift its output up or down, independent of its inputs. It helps the model fit the data better.

2.  **Activation Function:** The result ($z$) is then passed through a non-linear **activation function** ($\sigma$) to produce the neuron's final output.
    $$ \text{output} = \sigma(z) $$
    The non-linearity is crucial, as it allows the network to learn complex, non-linear relationships in the data. Without it, a deep network would be no more powerful than a single linear model.

## Layers

Neurons are organized into layers. A typical network has:
*   An **Input Layer:** Receives the raw data (e.g., the 28x28 pixels of an image).
*   **Hidden Layers:** One or more layers between the input and output. This is where most of the learning and feature extraction happens.
*   An **Output Layer:** Produces the final result (e.g., a probability for each of the 10 digits).

## Forward Propagation

This is the process of making a prediction. Input data is fed into the input layer, and the outputs of each layer are passed forward to become the inputs for the next layer, until the output layer is reached.

## Loss Function

Once the network makes a prediction, we need a way to measure how good (or bad) that prediction was. This is the job of the **loss function** (or cost function). It compares the network's prediction (`y_pred`) with the true target (`y_true`) and outputs a single scalar value representing the error.

*   A high loss means the prediction was poor.
*   A low loss means the prediction was good.

The goal of training is to adjust the weights and biases to minimize this loss value.

### Interpreting Loss Values

During training, you will observe the loss value decreasing over epochs. This is a good sign, indicating that your network is learning and improving its ability to make accurate predictions. However, the absolute value of the loss itself is often less important than its trend:

*   **Decreasing Loss:** The network is learning. This is what we want to see.
*   **Stagnant Loss:** The network might have stopped learning (e.g., due to a "dying ReLU" or a learning rate that is too small), or it might have reached a local minimum.
*   **Increasing Loss:** The network is diverging, meaning it's getting worse at its task. This often indicates a problem with the learning rate (too high) or the network architecture.
*   **Zero Loss:** While ideal, a loss of exactly zero, especially early in training, can sometimes indicate overfitting (the network has memorized the training data but won't generalize well to new data).

Monitoring the loss trend is crucial for debugging and understanding your model's training dynamics.

## Backpropagation and Gradient Descent

This is the core of the learning process.

1.  **Backpropagation:** After calculating the loss, we need to figure out how to adjust each weight and bias in the network to reduce the loss. Backpropagation is a mathematical algorithm that uses the **chain rule** from calculus to calculate the **gradient** (derivative) of the loss function with respect to every single weight and bias in the network.
    *   The **gradient** is a vector that points in the direction of the steepest increase of the loss function. Therefore, moving in the *opposite* direction of the gradient will cause the loss to decrease most quickly.

2.  **Gradient Descent:** This is the optimization algorithm that uses the gradients calculated by backpropagation to update the parameters. For each parameter ($\theta$), the update rule is:
    $$ \theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla_{\theta}L $$
    *   $\eta$ (eta) is the **learning rate**, a small hyperparameter that controls the step size of each update.
    *   $\nabla_{\theta}L$ is the gradient of the loss $L$ with respect to the parameter $\theta$.

In simple terms: Backpropagation figures out who to blame for the error (the gradients), and Gradient Descent updates the parameters based on that blame.

## Training Terminology

*   **Epoch:** One complete pass through the entire training dataset. During one epoch, the network sees every training sample once.
*   **Batch:** To make training more efficient and stable, we don't process the entire dataset at once. Instead, we divide it into smaller chunks called batches.
*   **Batch Size:** The number of training samples in one batch.
*   **Iteration:** The processing of one batch of data. This includes one forward pass and one backward pass, followed by a parameter update. If you have 60,000 samples and a batch size of 32, one epoch consists of $60000 / 32 = 1875$ iterations.
