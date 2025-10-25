import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.layers import Flatten, Layer
from tests.test_dense_layer import gradient_check # Reusing the gradient_check utility

# --- Test Flatten Layer ---
def test_flatten_layer():
    print("\n--- Testing Flatten Layer ---")
    batch_size = 2
    input_height = 3
    input_width = 4
    input_channels = 2
    input_shape = (batch_size, input_height, input_width, input_channels)

    flatten_layer = Flatten()
    input_data = np.random.randn(*input_shape)

    # Forward pass check
    output = flatten_layer.forward(input_data)
    expected_output_shape = (batch_size, input_height * input_width * input_channels)
    print(f"Flatten Layer Output Shape: {output.shape}")
    assert output.shape == expected_output_shape, "Flatten forward pass output shape mismatch!"
    assert np.allclose(output, input_data.reshape(batch_size, -1)), "Flatten forward pass values mismatch!"

    # Backward pass and gradient check
    gradient_check(flatten_layer, input_data.copy())


if __name__ == "__main__":
    test_flatten_layer()

