import numpy as np
import os

def preprocess_mnist_data():
    """
    Loads MNIST data, normalizes images, reshapes them, and one-hot encodes labels.
    Returns preprocessed training and test data.
    """
    npz_path = os.path.join("mnist_data", "mnist.npz")

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found. Please run mnist_loader.py first to generate the dataset.")
        return None, None

    with np.load(npz_path) as data:
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']

    # Normalize images to the range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape images to add a channel dimension (for CNNs)
    # Assuming (num_samples, height, width, channels) format
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # One-hot encode labels
    num_classes = 10 # Digits 0-9
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    print("MNIST data preprocessed successfully.")
    print(f"Preprocessed Training images shape: {train_images.shape}")
    print(f"Preprocessed Training labels shape: {train_labels.shape}")
    print(f"Preprocessed Test images shape: {test_images.shape}")
    print(f"Preprocessed Test labels shape: {test_labels.shape}")

    return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_mnist_data()
    if x_train is not None:
        # Optional: Save preprocessed data
        np.savez_compressed(
            os.path.join("mnist_data", "mnist_preprocessed.npz"),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        print("Preprocessed dataset saved to mnist_data/mnist_preprocessed.npz")
