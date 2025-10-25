import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.network import Network
from src.layers import Dense, ReLU
from src.losses import MeanSquaredError
import time

# --- Test Network Training ---
def test_network_training():
    print("\n--- Testing Network Training (XOR Problem) ---")
    
    # XOR problem data (2D arrays)
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Create a simple network
    net = Network()
    net.add(Dense(2, 3))
    net.add(ReLU())
    net.add(Dense(3, 1))

    # Compile the network with a loss function
    net.compile(MeanSquaredError())

    # Train the network
    start_time = time.time()
    net.train(X_train, y_train, epochs=1000, learning_rate=0.1, batch_size=1)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training took {training_time:.4f} seconds.")

    # Make predictions
    output = net.predict(X_train)
    print("\nPredictions after training:")
    for i in range(len(X_train)):
        print(f"Input: {X_train[i]}, Predicted: {output[i][0]:.4f}, True: {y_train[i][0]}")

    # Check if the loss has decreased (a simple sanity check)
    initial_loss = net.loss_function.loss(y_train, net.predict(X_train))
    # Train for a few more epochs
    net.train(X_train, y_train, epochs=10, learning_rate=0.1, batch_size=1)
    final_loss = net.loss_function.loss(y_train, net.predict(X_train))
    
    print(f"\nInitial Loss: {initial_loss:.4f}")
    print(f"Final Loss after more training: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease after further training."
    print("\nNetwork training test PASSED!")


if __name__ == "__main__":
    test_network_training()
