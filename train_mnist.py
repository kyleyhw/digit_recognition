import numpy as np
import matplotlib.pyplot as plt
import os

from src.network import Network
from src.layers import Convolutional, ReLU, MaxPooling, Flatten, Dense, Softmax
from src.losses import CategoricalCrossEntropy
from preprocess_mnist import preprocess_mnist_data

def build_cnn_model():
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
    net.compile(CategoricalCrossEntropy())
    return net

def plot_loss(epoch_losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plot_dir = "docs/images"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()

def main():
    print("--- Loading and Preprocessing MNIST Data ---")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = preprocess_mnist_data()
    
    # Create a subset for plotting purposes
    x_train_subset = x_train_full[:1000]
    y_train_subset = y_train_full[:1000]

    # Use a random subset of the test data for evaluation
    num_test_samples_to_evaluate = 200
    test_permutation = np.random.permutation(len(x_test_full))
    x_test_eval = x_test_full[test_permutation][:num_test_samples_to_evaluate]
    y_test_eval = y_test_full[test_permutation][:num_test_samples_to_evaluate]
    
    print(f"Full Training data shape: {x_train_full.shape}")
    print(f"Subset Training data shape: {x_train_subset.shape}")
    print(f"Evaluation Test data shape: {x_test_eval.shape}")

    # --- Generate Loss Plot for Subset Training (10 epochs) ---
    print("\n--- Training on Subset for Loss Plot (10 epochs) ---")
    net_subset = build_cnn_model()
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Ensure no old subset checkpoints interfere
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model_subset_plot_epoch_"):
            os.remove(os.path.join(checkpoint_dir, f))

    subset_epoch_losses = net_subset.train(
        x_train_subset, y_train_subset, epochs=10, learning_rate=0.01, batch_size=32,
        checkpoint_dir=checkpoint_dir, checkpoint_prefix="model_subset_plot", resume_from_checkpoint=False
    )
    plot_loss(subset_epoch_losses, 'Training Loss per Epoch (Subset Data)', 'training_loss_subset.png')

    # --- Continue Training on FULL Dataset (from epoch 6 to 10) ---
    print("\n--- Continuing Training on FULL Dataset (from epoch 6 to 10) ---")
    net_full = build_cnn_model()
    
    # Load the last known good full dataset checkpoint
    try:
        net_full.load_model(os.path.join(checkpoint_dir, "model_full_epoch_6.npz"))
        start_epoch_full = 6 # We finished epoch 6, so start from 6 for next epoch (7)
    except FileNotFoundError:
        print("ERROR: model_full_epoch_6.npz not found. Cannot resume full training.")
        print("Please ensure model_full_epoch_1.npz to model_full_epoch_6.npz exist in models/checkpoints/")
        return

    # Train for 4 more epochs (7, 8, 9, 10)
    full_epoch_losses = net_full.train(
        x_train_full, y_train_full, epochs=10, learning_rate=0.01, batch_size=32,
        checkpoint_dir=checkpoint_dir, checkpoint_prefix="model_full", resume_from_checkpoint=True
    )
    plot_loss(full_epoch_losses, 'Training Loss per Epoch (Full Data)', 'training_loss_full.png')

    # --- Save the Final Trained Model (Full Dataset) ---
    print("\n--- Saving the Final Trained Model (Full Dataset) ---")
    net_full.save_model("models/mnist_cnn_full_dataset.npz")

    # --- Evaluate the Network (Full Dataset Trained Model) ---
    print("\n--- Evaluating the Loaded Network ---")
    # Load the final full dataset model for evaluation
    final_eval_net = build_cnn_model()
    try:
        final_eval_net.load_model("models/mnist_cnn_full_dataset.npz")
    except FileNotFoundError:
        print("ERROR: Final model 'models/mnist_cnn_full_dataset.npz' not found for evaluation.")
        return

    total_correct = 0
    num_to_print = 20
    print_interval = len(x_test_eval) // num_to_print

    for i in range(len(x_test_eval)):
        x_sample = x_test_eval[i].reshape(1, 28, 28, 1)
        y_sample_true = y_test_eval[i]
        
        prediction_vector = final_eval_net.predict(x_sample)
        predicted_class = np.argmax(prediction_vector)
        true_class = np.argmax(y_sample_true)
        
        if predicted_class == true_class:
            total_correct += 1
            
        if i % print_interval == 0:
            print(f"Sample {i+1}: Predicted: {predicted_class}, True: {true_class}")

    accuracy = (total_correct / len(x_test_eval)) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()