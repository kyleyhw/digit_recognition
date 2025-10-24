import requests
import gzip
import numpy as np
import struct
import os

def download_mnist_data():
    """Downloads and extracts the MNIST dataset."""
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
        if not os.path.exists(path[:-3]): # Check if uncompressed file exists
            print(f"Downloading {url}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Extracting {path}...")
            with gzip.open(path, 'rb') as f_in:
                with open(path[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(path)


def load_mnist_data():
    """Loads the MNIST dataset from the downloaded files."""
    data_dir = "mnist_data"
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

    return (train_images, train_labels), (test_images, test_labels)


if __name__ == "__main__":
    npz_path = os.path.join("mnist_data", "mnist.npz")

    if os.path.exists(npz_path):
        print(f"Loading MNIST dataset from {npz_path}...")
        with np.load(npz_path) as data:
            train_images = data['train_images']
            train_labels = data['train_labels']
            test_images = data['test_images']
            test_labels = data['test_labels']
    else:
        print("MNIST dataset not found. Downloading and processing...")
        download_mnist_data()
        (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
        # Save as .npz file for easier loading next time
        np.savez_compressed(
            npz_path,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
        print(f"Dataset saved to {npz_path}")

    print("MNIST dataset loaded successfully.")
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
