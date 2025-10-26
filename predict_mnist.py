import numpy as np
import matplotlib.pyplot as plt
import os

from src.network import Network
from src.layers import Convolutional, ReLU, MaxPooling, Flatten, Dense, Softmax
from preprocess_mnist import preprocess_mnist_data

def visualize_predictions(x_test, y_test, net, num_samples=10, filepath="docs/images/prediction_examples.png"):
    """
    Visualizes a specified number of test samples with their true and predicted labels.
    """
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        x_sample = x_test[idx].reshape(1, 28, 28, 1)
        y_sample_true = np.argmax(y_test[idx])
        
        prediction_vector = net.predict(x_sample)
        predicted_class = np.argmax(prediction_vector)

        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        color = "green" if predicted_class == y_sample_true else "red"
        plt.title(f"True: {y_sample_true}\nPred: {predicted_class}", color=color)
        plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    print(f"Prediction examples plot saved to {filepath}")
    plt.close()

def main():
    print("--- Loading Test Data ---")
    # Load the preprocessed data
    (_, _), (x_test, y_test) = preprocess_mnist_data()
    
    print(f"Test data shape: {x_test.shape}")

    # --- Define the EXACT SAME CNN Architecture ---
    print("\n--- Building the CNN Model Structure ---")
    net = Network()
    net.add(Convolutional(input_shape=(28, 28, 1), num_filters=6, kernel_size=(5, 5), padding='same'))
    net.add(ReLU())
    net.add(MaxPooling(pool_size=(2, 2), stride=(2, 2)))
    net.add(Convolutional(input_shape=(14, 14, 6), num_filters=16, kernel_size=(5, 5), padding='valid'))
    net.add(ReLU())
    net.add(MaxPooling(pool_size=(2, 2), stride=(2, 2)))
    net.add(Flatten())
    net.add(Dense(400, 120))
    net.add(ReLU())
    net.add(Dense(120, 84))
    net.add(ReLU())
    net.add(Dense(84, 10))
    net.add(Softmax())
    print("Model structure built successfully.")

    # --- Load the Trained Model ---
    model_path = "models/mnist_cnn_subset_1000.npz"
    try:
        print(f"\n--- Loading the Trained Model from '{model_path}' ---")
        net.load_model(model_path)
    except FileNotFoundError:
        print(f"\nERROR: Model file '{model_path}' not found.")
        print("Please run 'python train_mnist.py' first to train and save the model.")
        return

    # --- Evaluate the Network ---
    print("\n--- Evaluating the Loaded Network ---")
    total_correct = 0
    num_to_print = 20
    print_interval = len(x_test) // num_to_print

    for i in range(len(x_test)):
        x_sample = x_test[i].reshape(1, 28, 28, 1)
        y_sample_true = y_test[i]
        
        prediction_vector = net.predict(x_sample)
        predicted_class = np.argmax(prediction_vector)
        true_class = np.argmax(y_sample_true)
        
        if predicted_class == true_class:
            total_correct += 1
            
        if i % print_interval == 0:
            print(f"Sample {i+1}: Predicted: {predicted_class}, True: {true_class}")

    accuracy = (total_correct / len(x_test)) * 100
    print(f"\nTest Accuracy from loaded model: {accuracy:.2f}%")

    # --- Visualize Predictions ---
    print("\n--- Visualizing Predictions ---")
    visualize_predictions(x_test, y_test, net, num_samples=10)

if __name__ == "__main__":
    main()
