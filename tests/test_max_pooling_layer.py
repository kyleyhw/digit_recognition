import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from layers import MaxPooling, Layer
from tests.test_dense_layer import gradient_check # Reusing the gradient_check utility
import time # For timing tests

# --- Test MaxPooling Layer ---
def test_max_pooling_layer():
    print("\n--- Testing MaxPooling Layer ---")
    batch_size = 2
    in_height, in_width, in_channels = 4, 4, 1
    pool_size = (2, 2)
    stride = (2, 2)

    input_shape = (in_height, in_width, in_channels)

    max_pooling_layer = MaxPooling(pool_size, stride)
    input_data = np.array([
        [
            [[1], [2], [3], [4]],
            [[5], [6], [7], [8]],
            [[9], [10], [11], [12]],
            [[13], [14], [15], [16]]
        ],
        [
            [[10], [20], [30], [40]],
            [[50], [60], [70], [80]],
            [[90], [100], [110], [120]],
            [[130], [140], [150], [160]]
        ]
    ], dtype=float)
    input_data = input_data.reshape(batch_size, in_height, in_width, in_channels)

    print(f"Test Input Data:\n{input_data}")

    # Forward pass check
    start_time = time.time()
    output = max_pooling_layer.forward(input_data)
    end_time = time.time()
    forward_time = end_time - start_time

    expected_output_shape = (batch_size, in_height // pool_size[0], in_width // pool_size[1], in_channels)
    expected_output = np.array([
        [
            [[6], [8]],
            [[14], [16]]
        ],
        [
            [[60], [80]],
            [[140], [160]]
        ]
    ], dtype=float).reshape(batch_size, in_height // pool_size[0], in_width // pool_size[1], in_channels)

    print(f"Expected Output:\n{expected_output}")
    print(f"MaxPooling Layer Output:\n{output}")
    print(f"MaxPooling Layer Output Shape: {output.shape}")
    assert output.shape == expected_output_shape, "MaxPooling forward pass output shape mismatch!"
    assert np.allclose(output, expected_output), "MaxPooling forward pass values mismatch!"
    print(f"Forward pass took {forward_time:.4f} seconds.")

    # Backward pass and gradient check
    # Use a smaller random input for gradient check
    gc_batch_size = 1
    gc_in_height, gc_in_width, gc_in_channels = 3, 3, 1
    gc_pool_size = (2, 2)
    gc_stride = (1, 1)

    gc_max_pooling_layer = MaxPooling(gc_pool_size, gc_stride)
    gc_input_data = np.random.randn(gc_batch_size, gc_in_height, gc_in_width, gc_in_channels)

    start_time = time.time()
    gradient_check(gc_max_pooling_layer, gc_input_data.copy())
    end_time = time.time()
    backward_time = end_time - start_time
    print(f"Backward pass (gradient check) took {backward_time:.4f} seconds.")


if __name__ == "__main__":
    test_max_pooling_layer()