import tkinter as tk
from tkinter import Canvas, Label, Button
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.network import Network
from src.layers import Convolutional, ReLU, MaxPooling, Flatten, Dense, Softmax

class DigitRecognizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Recognizer")

        self.canvas_width = 280
        self.canvas_height = 280
        self.pen_width = 15 # Make the drawing thick

        self.canvas = Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        self.label = Label(master, text="Draw a digit and press Predict", font=("Helvetica", 16))
        self.label.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.clear_button = Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)

        # In-memory image for drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Load the neural network
        self.load_network()

    def load_network(self):
        print("--- Building the CNN Model Structure ---")
        self.net = Network()
        self.net.add(Convolutional(input_shape=(28, 28, 1), num_filters=6, kernel_size=(5, 5), padding='same'))
        self.net.add(ReLU())
        self.net.add(MaxPooling(pool_size=(2, 2), stride=(2, 2)))
        self.net.add(Convolutional(input_shape=(14, 14, 6), num_filters=16, kernel_size=(5, 5), padding='valid'))
        self.net.add(ReLU())
        self.net.add(MaxPooling(pool_size=(2, 2), stride=(2, 2)))
        self.net.add(Flatten())
        self.net.add(Dense(400, 120))
        self.net.add(ReLU())
        self.net.add(Dense(120, 84))
        self.net.add(ReLU())
        self.net.add(Dense(84, 10))
        self.net.add(Softmax())
        print("Model structure built successfully.")

        try:
            print("\n--- Loading the Trained Model from 'models/mnist_cnn_subset_1000.npz' ---")
            self.net.load_model("models/mnist_cnn_subset_1000.npz")
        except FileNotFoundError:
            self.label.config(text="ERROR: Model file not found!")
            print("\nERROR: Model file 'models/mnist_cnn_subset_1000.npz' not found.")
            print("Please run 'python train_mnist.py' first to train and save the model.")

    def paint(self, event):
        x1, y1 = (event.x - self.pen_width), (event.y - self.pen_width)
        x2, y2 = (event.x + self.pen_width), (event.y + self.pen_width)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        # Draw on the in-memory image as well
        self.draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def clear_canvas(self):
        self.canvas.delete("all")
        # Clear the in-memory image
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill='black')
        self.label.config(text="Canvas cleared. Draw a digit.")

    def predict_digit(self):
        # Resize the image to 28x28
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        
        # Invert colors (black background, white digit)
        img_inverted = ImageOps.invert(img_resized)

        # Convert to numpy array and normalize
        img_array = np.array(img_inverted).astype('float32') / 255.0

        # Reshape for the model
        img_reshaped = img_array.reshape(1, 28, 28, 1)

        # Make a prediction
        prediction_vector = self.net.predict(img_reshaped)
        predicted_class = np.argmax(prediction_vector)

        self.label.config(text=f"Predicted Digit: {predicted_class}")
        print(f"Prediction: {predicted_class}, Raw Vector: {prediction_vector}")

if __name__ == '__main__':
    root = tk.Tk()
    gui = DigitRecognizerGUI(root)
    root.mainloop()
