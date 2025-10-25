import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from layers import Dense, Layer # Assuming Layer is also in layers.py

# --- Gradient Checking Utility ---
def gradient_check(layer, input_data_original, epsilon=1e-4):
    """
    Performs gradient checking for a layer's backward pass.
    Compares analytical gradients (from layer.backward) with numerical gradients.
    Uses a simple squared loss: L = sum(output**2)
    """
    print(f"\n--- Gradient Check for {layer.__class__.__name__} ---")
    is_correct = True
    learning_rate = 0.01 # Dummy learning rate for backward pass

    # Store initial state for restoration
    initial_weights = layer.weights.copy() if hasattr(layer, 'weights') else None
    initial_bias = layer.bias.copy() if hasattr(layer, 'bias') else None

    # --- Analytical Gradient Calculation ---
    # Perform a forward pass to get the output and set layer.input
    layer.forward(input_data_original.copy())
    analytical_output_of_layer = layer.output.copy()
    analytical_output_gradient_to_layer = 2 * analytical_output_of_layer # dL/d(output) = 2 * output for L=sum(output**2)

    # Call backward to get analytical input gradient and store parameter gradients
    analytical_input_gradient = layer.backward(analytical_output_gradient_to_layer, learning_rate=0) # learning_rate=0 to prevent updates during check

    # Capture analytical parameter gradients (if they exist and were stored by backward)
    analytical_weights_gradient = layer.weights_gradient.copy() if hasattr(layer, 'weights_gradient') else None
    analytical_bias_gradient = layer.bias_gradient.copy() if hasattr(layer, 'bias_gradient') else None

    # Restore initial weights and bias for numerical checking
    if hasattr(layer, 'weights'):
        layer.weights = initial_weights.copy()
    if hasattr(layer, 'bias'):
        layer.bias = initial_bias.copy()


    # --- Numerical Gradient Calculation ---

    # 1. Numerical Gradient for Input
    numerical_input_gradient = np.zeros_like(input_data_original)
    input_data_perturbed = input_data_original.copy()

    for i in range(input_data_original.shape[0]):
        for j in range(input_data_original.shape[1]):
            original_value = input_data_original[i, j]

            # Perturb +epsilon
            input_data_perturbed[i, j] = original_value + epsilon
            output_plus = layer.forward(input_data_perturbed)
            loss_plus = np.sum(output_plus**2)

            # Perturb -epsilon
            input_data_perturbed[i, j] = original_value - epsilon
            output_minus = layer.forward(input_data_perturbed)
            loss_minus = np.sum(output_minus**2)

            # Restore original value
            input_data_perturbed[i, j] = original_value

            numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
            numerical_input_gradient[i, j] = numerical_gradient

    # Compare input gradients
    relative_diff_input = np.max(np.abs(analytical_input_gradient - numerical_input_gradient) /
                                 (np.abs(analytical_input_gradient) + np.abs(numerical_input_gradient) + 1e-8))
    print(f"Input Gradient Relative Difference: {relative_diff_input}")
    if relative_diff_input > 1e-5:
        print("WARNING: Input gradient check failed!")
        is_correct = False


    # 2. Numerical Gradient for Weights (if layer has them)
    if hasattr(layer, 'weights') and analytical_weights_gradient is not None:
        numerical_weights_gradient = np.zeros_like(initial_weights)
        
        for i in range(initial_weights.shape[0]):
            for j in range(initial_weights.shape[1]):
                original_value = initial_weights[i, j]

                # Perturb +epsilon
                layer.weights[i, j] = original_value + epsilon
                output_plus = layer.forward(input_data_original) # Use original input for forward pass
                loss_plus = np.sum(output_plus**2)

                # Perturb -epsilon
                layer.weights[i, j] = original_value - epsilon
                output_minus = layer.forward(input_data_original)
                loss_minus = np.sum(output_minus**2)

                # Restore original value
                layer.weights[i, j] = original_value

                numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
                numerical_weights_gradient[i, j] = numerical_gradient

        relative_diff_weights = np.max(np.abs(analytical_weights_gradient - numerical_weights_gradient) /
                                       (np.abs(analytical_weights_gradient) + np.abs(numerical_weights_gradient) + 1e-8))
        print(f"Weights Gradient Relative Difference: {relative_diff_weights}")
        if relative_diff_weights > 1e-5:
            print("WARNING: Weights gradient check failed!")
            is_correct = False
        layer.weights = initial_weights.copy() # Restore weights after checking


    # 3. Numerical Gradient for Bias (if layer has them)
    if hasattr(layer, 'bias') and analytical_bias_gradient is not None:
        numerical_bias_gradient = np.zeros_like(initial_bias)

        for j in range(initial_bias.shape[0]):
            original_value = initial_bias[j]

            # Perturb +epsilon
            layer.bias[j] = original_value + epsilon
            output_plus = layer.forward(input_data_original)
            loss_plus = np.sum(output_plus**2)

            # Perturb -epsilon
            layer.bias[j] = original_value - epsilon
            output_minus = layer.forward(input_data_original)
            loss_minus = np.sum(output_minus**2)

            # Restore original value
            layer.bias[j] = original_value

            numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
            numerical_bias_gradient[j] = numerical_gradient

        relative_diff_bias = np.max(np.abs(analytical_bias_gradient - numerical_bias_gradient) /
                                    (np.abs(analytical_bias_gradient) + np.abs(numerical_bias_gradient) + 1e-8))
        print(f"Bias Gradient Relative Difference: {relative_diff_bias}")
        if relative_diff_bias > 1e-5:
            print("WARNING: Bias gradient check failed!")
            is_correct = False
        layer.bias = initial_bias.copy() # Restore bias after checking


    if is_correct:
        print(f"Gradient Check for {layer.__class__.__name__} PASSED!")
    else:
        print(f"Gradient Check for {layer.__class__.__name__} FAILED!")
    return is_correct


# --- Test Dense Layer ---
def test_dense_layer():
    print("\n--- Testing Dense Layer ---")
    input_size = 3
    output_size = 2
    batch_size = 4
    learning_rate = 0.01

    dense_layer = Dense(input_size, output_size)
    input_data = np.random.randn(batch_size, input_size)
    # output_gradient is no longer needed here as gradient_check generates it
    
    # Forward pass check (simple inspection)
    output = dense_layer.forward(input_data)
    print(f"Dense Layer Output Shape: {output.shape}")
    assert output.shape == (batch_size, output_size)

    # Backward pass and gradient check
    gradient_check(dense_layer, input_data.copy()) # Removed output_gradient


if __name__ == "__main__":
    test_dense_layer()
