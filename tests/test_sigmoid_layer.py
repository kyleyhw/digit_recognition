import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.layers import Sigmoid, Layer
from tests.test_dense_layer import gradient_check # Reusing the gradient_check utility
import time # For timing tests

# --- Test Sigmoid Layer ---
def test_sigmoid_layer():
    print("\n--- Testing Sigmoid Layer ---")
    input_shape = (4, 5) # batch_size, input_features
    learning_rate = 0.01

    sigmoid_layer = Sigmoid()
    input_data = np.array([
        [-1.0, 0.0, 1.0, 5.0, -5.0],
        [0.5, -0.5, 2.0, -2.0, 0.0],
        [1.0, 1.0, -1.0, 2.0, 0.0],
        [-0.5, 0.0, 0.5, -1.0, 1.5]
    ])
    print(f"Test Input Data:\n{input_data}")

    # Forward pass check (simple inspection)
    start_time = time.time()
    output = sigmoid_layer.forward(input_data)
    end_time = time.time()
    forward_time = end_time - start_time
    expected_output = 1 / (1 + np.exp(-input_data))
    print(f"Expected Output:\n{expected_output}")
    print(f"Sigmoid Layer Output:\n{output}")
    assert np.allclose(output, expected_output), "Sigmoid forward pass failed!"
    print(f"Forward pass took {forward_time:.4f} seconds.")

    # Backward pass and gradient check
    start_time = time.time()
    gradient_check(sigmoid_layer, input_data.copy())
    end_time = time.time()
    backward_time = end_time - start_time
    print(f"Backward pass (gradient check) took {backward_time:.4f} seconds.")


if __name__ == "__main__":
    test_sigmoid_layer()

