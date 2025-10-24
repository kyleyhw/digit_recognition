import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the mnist_loader script has been run to generate mnist.npz
# If not, you might want to run it first: python mnist_loader.py

def visualize_mnist_samples(num_samples=10):
    """
    Loads MNIST data from mnist.npz and visualizes a specified number of samples.
    """
    npz_path = os.path.join("mnist_data", "mnist.npz")

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found. Please run mnist_loader.py first to generate the dataset.")
        return

    with np.load(npz_path) as data:
        train_images = data['train_images']
        train_labels = data['train_labels']

    # Select random samples
    indices = np.random.choice(len(train_images), num_samples, replace=False)

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(train_images[idx], cmap='gray')
        plt.title(f"Label: {train_labels[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_mnist_samples(num_samples=10)
