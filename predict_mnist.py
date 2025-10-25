import numpy as np
from src.network import Network
from src.layers import Convolutional, ReLU, MaxPooling, Flatten, Dense, Softmax
from preprocess_mnist import preprocess_mnist_data

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
    try:
        print("\n--- Loading the Trained Model from 'mnist_cnn.npz' ---")
        net.load_model("models/mnist_cnn_subset_1000.npz")
    except FileNotFoundError:
        print("\nERROR: Model file 'mnist_cnn.npz' not found.")
        print("Please run 'python train_mnist.py' first to train and save the model.")
        return

    # --- Evaluate the Network ---
    print("\n--- Evaluating the Loaded Network ---")
    total_correct = 0
    for i in range(len(x_test)):
        x_sample = x_test[i].reshape(1, 28, 28, 1)
        y_sample_true = y_test[i]
        
        prediction_vector = net.predict(x_sample)
        predicted_class = np.argmax(prediction_vector)
        true_class = np.argmax(y_sample_true)
        
        if predicted_class == true_class:
            total_correct += 1
            
        if i < 10 or i > len(x_test) - 10:
            print(f"Sample {i+1}: Predicted: {predicted_class}, True: {true_class}")

    accuracy = (total_correct / len(x_test)) * 100
    print(f"\nTest Accuracy from loaded model: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
