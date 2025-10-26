import numpy as np
from tqdm import tqdm # Import tqdm
# Assuming layers.py is in the same directory, we'll import necessary classes
from .layers import Layer # We'll need this for type hinting or just general understanding
from .losses import Loss # Assuming losses.py is in the same directory

class Network:
    """
    A sequential neural network model.
    This class will orchestrate the forward and backward passes through its layers.
    """
    def __init__(self):
        self.layers = []
        self.loss_function = None # To be set by the user

    def add(self, layer):
        """
        Adds a layer to the network.
        """
        self.layers.append(layer)

    def compile(self, loss_function):
        """
        Configures the model for training.
        """
        self.loss_function = loss_function

    def forward(self, input_data):
        """
        Performs the forward pass through all layers in the network.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        """
        Performs the backward pass through all layers in reverse order.
        """
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, X_train, y_train, epochs, learning_rate, batch_size=32):
        """
        Orchestrates the training loop.
        """
        if self.loss_function is None:
            raise ValueError("Loss function not compiled. Call network.compile(loss_function) first.")

        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            total_loss = 0
            # Shuffle data for each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            # Use tqdm for a progress bar with ETA
            with tqdm(range(0, num_samples, batch_size), unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for i in pbar:
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]

                    # Forward pass
                    output = self.forward(X_batch)

                    # Calculate loss
                    loss = self.loss_function.loss(y_batch, output)
                    total_loss += loss

                    # Backward pass
                    output_gradient = self.loss_function.prime(y_batch, output)
                    self.backward(output_gradient, learning_rate)
                    
                    # Update progress bar description with current loss
                    pbar.set_postfix(loss=f"{loss:.4f}")
            
            avg_loss = total_loss / (num_samples / batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def predict(self, input_data):
        """
        Makes predictions using the trained network.
        """
        return self.forward(input_data)

    def save_model(self, filepath):
        """
        Saves the weights and biases of all layers with parameters to a .npz file.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                params[f'weights_{i}'] = layer.weights
            if hasattr(layer, 'bias'):
                params[f'bias_{i}'] = layer.bias
        
        np.savez_compressed(filepath, **params)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Loads weights and biases from a .npz file into the network layers.
        The network architecture must match the saved model.
        """
        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = data[f'weights_{i}']
            if hasattr(layer, 'bias'):
                layer.bias = data[f'bias_{i}']
        print(f"Model loaded from {filepath}")
