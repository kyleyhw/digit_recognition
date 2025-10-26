import numpy as np
from tqdm import tqdm # Import tqdm
import os # Import os for path manipulation
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

    def train(self, X_train, y_train, epochs, learning_rate, batch_size=32, checkpoint_dir=None, resume_from_checkpoint=False):
        """
        Orchestrates the training loop.
        Returns a list of average losses for each epoch.
        """
        if self.loss_function is None:
            raise ValueError("Loss function not compiled. Call network.compile(loss_function) first.")

        num_samples = X_train.shape[0]
        epoch_losses = [] # List to store average loss per epoch
        start_epoch = 0

        if resume_from_checkpoint and checkpoint_dir:
            latest_checkpoint = self._find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print(f"Resuming training from checkpoint: {latest_checkpoint}")
                self.load_model(latest_checkpoint)
                start_epoch = int(latest_checkpoint.split('epoch_')[1].split('.')[0])
                # Optionally, load previous epoch_losses if saved with checkpoint
                # For simplicity, we'll start collecting losses from resume point
            else:
                print("No checkpoint found to resume from. Starting new training.")

        for epoch in range(start_epoch, epochs):
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
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.npz")
                self.save_model(checkpoint_path)
        
        return epoch_losses # Return the collected losses

    def _find_latest_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            return None
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.npz')]
        if not checkpoints:
            return None
        
        # Sort by epoch number to find the latest
        checkpoints.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
        return os.path.join(checkpoint_dir, checkpoints[-1])

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
