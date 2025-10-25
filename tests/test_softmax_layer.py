import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from layers import Softmax, Layer
from tests.test_dense_layer import gradient_check # Reusing the gradient_check utility

# --- Test Softmax Layer ---
def test_softmax_layer():
    print("\n--- Testing Softmax Layer ---")
    input_shape = (4, 5) # batch_size, input_features
    learning_rate = 0.01

    softmax_layer = Softmax()
    input_data = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
        [10.0, 1.0, 0.0, -1.0, -10.0]
    ])
    # Forward pass check (simple inspection)
    output = softmax_layer.forward(input_data)
    exp_values = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
    expected_output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    print(f"Softmax Layer Output:\n{output}")
    assert np.allclose(output, expected_output), "Softmax forward pass failed!"

    # Backward pass and gradient check
    # Note: Gradient checking for Softmax is tricky when not combined with Cross-Entropy Loss.
    # The gradient_check utility uses a simple squared loss, which might not be ideal for Softmax.
    # However, it will still provide a basic sanity check.
    gradient_check(softmax_layer, input_data.copy(), epsilon=1e-3) # Increased epsilon for Softmax stability


if __name__ == "__main__":
    test_softmax_layer()
