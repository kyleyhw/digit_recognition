import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.layers import ReLU, Layer
from tests.test_dense_layer import gradient_check # Reusing the gradient_check utility
import time # For timing tests

# --- Test ReLU Layer ---
def test_relu_layer():
    print("\n--- Testing ReLU Layer ---")
    input_shape = (4, 5) # batch_size, input_features
    learning_rate = 0.01

    relu_layer = ReLU()
    # Input data with positive and negative values to test ReLU behavior
    # Input data with positive and negative values to test ReLU behavior, avoiding exact zeros
    input_data = np.random.randn(*input_shape) * 2 # Scale to get some values > 0 and < 0
    print(f"Test Input Data:\n{input_data}")

    # Forward pass check (output shape)
    start_time = time.time()
    output = relu_layer.forward(input_data)
    end_time = time.time()
    forward_time = end_time - start_time
    print(f"ReLU Layer Output:\n{output}")
    print(f"ReLU Layer Output Shape: {output.shape}")
    assert output.shape == input_shape, "ReLU forward pass output shape mismatch!"
    print(f"Forward pass took {forward_time:.4f} seconds.")

    # Backward pass and gradient check
    start_time = time.time()
    gradient_check(relu_layer, input_data.copy())
    end_time = time.time()
    backward_time = end_time - start_time
    print(f"Backward pass (gradient check) took {backward_time:.4f} seconds.")


if __name__ == "__main__":
    test_relu_layer()
