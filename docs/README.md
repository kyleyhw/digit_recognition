# Documentation Hub

Welcome to the documentation for the "CNN from Scratch" project. This guide is designed to walk you through the fundamental theory and implementation of a convolutional neural network, assuming no prior background in machine learning.

Our goal is to demystify the core components of a neural network by building them from the ground up, connecting the mathematical theory directly to the Python code.

## Table of Contents

Follow these documents in order for a comprehensive understanding.

1.  **[Fundamental Concepts](./Concepts.md)**
    *   Start here if you are new to machine learning. This document explains the core ideas and terminology, such as what a neural network is, what a "loss function" means, and the basic idea behind "training."

2.  **[The Layers Explained](./Layers_Explained.md)**
    *   This is a deep dive into each type of layer we built. For each one, we explain its purpose, the mathematics of its forward and backward passes, and show how the math is translated directly into the code in `src/layers.py`.

3.  **[Loss Functions](./loss_functions.md)**
    *   Learn how we measure the network's performance. This document details the Mean Squared Error and Categorical Cross-Entropy loss functions, including their mathematical derivations and code implementation.

4.  **[The Network Class](./network_class.md)**
    *   Understand the orchestrator of the whole process. This document explains how the `Network` class holds the layers together and manages the training and prediction logic.

5.  **[Training the CNN on MNIST](./train_cnn.md)**
    *   See how all the pieces come together. This document breaks down the final `train_mnist.py` script, explaining the specific architecture of our digit-recognizing CNN and how we train and evaluate it.
