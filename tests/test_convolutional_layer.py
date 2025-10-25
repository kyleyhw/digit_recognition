import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.layers import Convolutional, Layer
from tests.test_dense_layer import gradient_check # Reusing the gradient_check utility

import time # For timing tests

# --- Test Convolutional Layer ---
def test_convolutional_layer():
    print("\n--- Testing Convolutional Layer ---")
    batch_size = 2
    in_height, in_width, in_channels = 10, 10, 3
    num_filters = 4
    k_height, k_width = 3, 3
    stride = (1, 1)
    padding = 'valid'
    learning_rate = 0.01

    input_shape = (in_height, in_width, in_channels)
    kernel_size = (k_height, k_width)

    conv_layer = Convolutional(input_shape, num_filters, kernel_size, stride, padding)
    input_data = np.random.randn(batch_size, in_height, in_width, in_channels)
    print(f"Test Input Data Shape: {input_data.shape}")

    # Forward pass check (shape)
    start_time = time.time()
    output = conv_layer.forward(input_data)
    end_time = time.time()
    forward_time = end_time - start_time
    expected_out_height = (in_height - k_height) // stride[0] + 1
    expected_out_width = (in_width - k_width) // stride[1] + 1
    expected_output_shape = (batch_size, expected_out_height, expected_out_width, num_filters)
    print(f"Convolutional Layer Output Shape: {output.shape}")
    assert output.shape == expected_output_shape, "Convolutional forward pass output shape mismatch!"
    print(f"Forward pass took {forward_time:.4f} seconds.")

    # Test with 'same' padding
    conv_layer_same = Convolutional(input_shape, num_filters, kernel_size, stride=(1,1), padding='same')
    start_time_same = time.time()
    output_same = conv_layer_same.forward(input_data)
    end_time_same = time.time()
    forward_time_same = end_time_same - start_time_same
    expected_output_shape_same = (batch_size, in_height, in_width, num_filters)
    print(f"Convolutional Layer (same padding) Output Shape: {output_same.shape}")
    assert output_same.shape == expected_output_shape_same, "Convolutional forward pass (same padding) output shape mismatch!"
    print(f"Forward pass (same padding) took {forward_time_same:.4f} seconds.")


    # Backward pass and gradient check
    # Note: Gradient checking for Convolutional layers can be very slow due to iterating over all elements.
    # We'll use a smaller input for the gradient check.
    print("\n--- Running Gradient Check for Convolutional Layer (smaller input) ---")
    gc_batch_size = 1
    gc_in_height, gc_in_width, gc_in_channels = 5, 5, 2
    gc_num_filters = 3
    gc_k_height, gc_k_width = 3, 3
    gc_stride = (1, 1)
    gc_padding = 'valid'

    gc_input_shape = (gc_in_height, gc_in_width, gc_in_channels)
    gc_kernel_size = (gc_k_height, gc_k_width)

    gc_conv_layer = Convolutional(gc_input_shape, gc_num_filters, gc_kernel_size, gc_stride, gc_padding)
    gc_input_data = np.random.randn(gc_batch_size, gc_in_height, gc_in_width, gc_in_channels)

    start_time_gc = time.time()
    gradient_check(gc_conv_layer, gc_input_data.copy())
    end_time_gc = time.time()
    backward_time = end_time_gc - start_time_gc
    print(f"Backward pass (gradient check) took {backward_time:.4f} seconds.")


if __name__ == "__main__":
    test_convolutional_layer()
