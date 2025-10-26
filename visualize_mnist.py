import matplotlib.pyplot as plt
import numpy as np
import os

# Assuming preprocess_mnist.py has been run to generate mnist_preprocessed.npz
from preprocess_mnist import preprocess_mnist_data

def visualize_mnist_samples(num_samples=10, filepath="docs/images/sample_data.png"):
    """
    Loads preprocessed MNIST data and visualizes a specified number of samples.
    """
    # Load the preprocessed data
    (x_train, y_train), (x_test, y_test) = preprocess_mnist_data()

    # Reshape images from (N, 28, 28, 1) to (N, 28, 28) for imshow
    x_train_reshaped = x_train.reshape(-1, 28, 28)
    y_train_labels = np.argmax(y_train, axis=1)

    # Select random samples from training data
    indices = np.random.choice(len(x_train_reshaped), num_samples, replace=False)

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(x_train_reshaped[idx], cmap='gray')
        plt.title(f"Label: {y_train_labels[idx]}")
        plt.axis('off')
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    print(f"Sample data plot saved to {filepath}")
    plt.close() # Close the plot to free memory

if __name__ == "__main__":
    visualize_mnist_samples(num_samples=10)
