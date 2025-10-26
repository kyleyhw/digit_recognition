import numpy as np
from src.network import Network
from src.layers import Convolutional, ReLU, MaxPooling, Flatten, Dense, Softmax
from src.losses import CategoricalCrossEntropy
from preprocess_mnist import preprocess_mnist_data # Assuming this function is available

def main():
    print("--- Loading and Preprocessing MNIST Data ---")
    # Load the preprocessed data
    (x_train, y_train), (x_test, y_test) = preprocess_mnist_data()
    
    # For faster development, let's use a subset of the data
    # Remove these lines to train on the full dataset
    x_train = x_train
    y_train = y_train
    x_test = x_test
    y_test = y_test
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # --- Define the CNN Architecture ---
    print("\n--- Building the CNN Model ---")
    net = Network()
    
    # Layer 1: Convolutional -> ReLU -> MaxPooling
    net.add(Convolutional(input_shape=(28, 28, 1), num_filters=6, kernel_size=(5, 5), padding='same'))
    net.add(ReLU())
    net.add(MaxPooling(pool_size=(2, 2), stride=(2, 2)))
    
    # Layer 2: Convolutional -> ReLU -> MaxPooling
    # The input shape for this layer is the output of the previous MaxPooling layer
    net.add(Convolutional(input_shape=(14, 14, 6), num_filters=16, kernel_size=(5, 5), padding='valid'))
    net.add(ReLU())
    net.add(MaxPooling(pool_size=(2, 2), stride=(2, 2)))
    
    # Layer 3: Flatten the output for the Dense layers
    net.add(Flatten())
    
    # Layer 4: Dense -> ReLU
    # The input size for this Dense layer is the flattened output of the previous MaxPooling layer
    # (5*5*16 = 400)
    net.add(Dense(400, 120))
    net.add(ReLU())
    
    # Layer 5: Dense -> ReLU
    net.add(Dense(120, 84))
    net.add(ReLU())
    
    # Layer 6: Output Layer (Dense -> Softmax)
    net.add(Dense(84, 10))
    net.add(Softmax())

    # Compile the network with Categorical Cross-Entropy loss
    net.compile(CategoricalCrossEntropy())
    print("Model built successfully.")

    # --- Train the Network ---
    print("\n--- Training the Network ---")
    net.train(x_train, y_train, epochs=10, learning_rate=0.01, batch_size=32)

    # --- Save the Trained Model ---
    print("\n--- Saving the Trained Model ---")
    net.save_model("models/mnist_cnn_full_set.npz")

    # --- Evaluate the Network ---
    print("\n--- Evaluating the Network ---")
    total_correct = 0
    num_to_print = 20 # Number of evenly spaced samples to print
    print_interval = len(x_test) // num_to_print

    for i in range(len(x_test)):
        x_sample = x_test[i].reshape(1, 28, 28, 1)
        y_sample_true = y_test[i]
        
        # Make a prediction
        prediction_vector = net.predict(x_sample)
        
        # Get the predicted class index
        predicted_class = np.argmax(prediction_vector)
        true_class = np.argmax(y_sample_true)
        
        if predicted_class == true_class:
            total_correct += 1
            
        # Print evenly spaced samples
        if i % print_interval == 0:
            print(f"Sample {i+1}: Predicted: {predicted_class}, True: {true_class}")

    accuracy = (total_correct / len(x_test)) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
