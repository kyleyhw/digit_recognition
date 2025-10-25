import numpy as np
import os
import requests
import gzip
import struct

def download_mnist_data():
    """Downloads and extracts the MNIST dataset if not already present."""
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    data_dir = "mnist_data"
    os.makedirs(data_dir, exist_ok=True)

    for file in files:
        url = base_url + file
        path = os.path.join(data_dir, file)
        uncompressed_path = path[:-3]

        if os.path.exists(uncompressed_path):
            continue

        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Extracting {path}...")
        with gzip.open(path, 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(path)

def create_mnist_npz():
    """Loads the raw MNIST files and saves them into a single .npz file."""
    data_dir = "mnist_data"
    npz_path = os.path.join(data_dir, "mnist.npz")

    if os.path.exists(npz_path):
        return

    print("Creating mnist.npz from raw files...")
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    with open(train_labels_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

    with open(test_labels_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

    with open(train_images_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(len(train_labels), rows, cols)

    with open(test_images_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(len(test_labels), rows, cols)

    np.savez_compressed(
        npz_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    print(f"Dataset saved to {npz_path}")


def preprocess_mnist_data():
    """
    Loads MNIST data, normalizes images, reshapes them, and one-hot encodes labels.
    Returns preprocessed training and test data.
    """
    npz_path = os.path.join("mnist_data", "mnist.npz")

    # Ensure raw data is downloaded and mnist.npz is created
    download_mnist_data()
    create_mnist_npz()

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
    preprocessed_npz_path = os.path.join("mnist_data", "mnist_preprocessed.npz")
    if os.path.exists(preprocessed_npz_path):
        print(f"'{preprocessed_npz_path}' already exists. Skipping preprocessing.")
    else:
        print("Preprocessing data...")
        (x_train, y_train), (x_test, y_test) = preprocess_mnist_data()
        if x_train is not None:
            # Save preprocessed data
            np.savez_compressed(
                preprocessed_npz_path,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            print(f"Preprocessed dataset saved to {preprocessed_npz_path}")
