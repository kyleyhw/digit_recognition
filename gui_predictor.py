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

        self.rows = 28
        self.cols = 28
        self.pixel_size = 10
        self.canvas_width = self.cols * self.pixel_size
        self.canvas_height = self.rows * self.pixel_size

        self.canvas = Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        self.label = Label(master, text="Draw a digit and press Predict", font=("Helvetica", 16))
        self.label.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.clear_button = Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.pixel_grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.image_array = np.zeros((self.rows, self.cols))
        self.create_grid()

        self.load_network()

    def create_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.pixel_size
                y1 = r * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.pixel_grid[r][c] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")

    def paint(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        if 0 <= col < self.cols and 0 <= row < self.rows:
            # Paint the main cell and surrounding cells for a thicker line
            for r in range(max(0, row-1), min(self.rows, row+2)):
                for c in range(max(0, col-1), min(self.cols, col+2)):
                    self.canvas.itemconfig(self.pixel_grid[r][c], fill="white")
                    self.image_array[r, c] = 1.0

    def clear_canvas(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.canvas.itemconfig(self.pixel_grid[r][c], fill="black")
        self.image_array = np.zeros((self.rows, self.cols))
        self.label.config(text="Canvas cleared. Draw a digit.")

    def process_image_for_prediction(self):
        # Find bounding box of the digit
        rows = np.any(self.image_array, axis=1)
        cols = np.any(self.image_array, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None # Return None if canvas is empty
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        digit_image = self.image_array[rmin:rmax+1, cmin:cmax+1]

        # Get dimensions and calculate padding
        h, w = digit_image.shape
        delta = abs(h - w)
        if h > w:
            pad_left, pad_right = delta // 2, delta - (delta // 2)
            pad_top, pad_bottom = 0, 0
        else:
            pad_top, pad_bottom = delta // 2, delta - (delta // 2)
            pad_left, pad_right = 0, 0

        # Add padding to make it square
        digit_padded = np.pad(digit_image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')

        # Resize to 20x20 and add padding to make it 28x28 (to better match MNIST)
        img = Image.fromarray((digit_padded * 255).astype(np.uint8))
        img = img.resize((20, 20), Image.LANCZOS)
        img_padded = np.pad(np.array(img), ((4, 4), (4, 4)), 'constant')

        # Normalize and reshape for the model
        img_final = img_padded.astype('float32') / 255.0
        return img_final.reshape(1, 28, 28, 1)

    def predict_digit(self):
        processed_image = self.process_image_for_prediction()

        if processed_image is None:
            self.label.config(text="Canvas is empty!")
            return

        # Make a prediction
        prediction_vector = self.net.predict(processed_image)
        predicted_class = np.argmax(prediction_vector)

        self.label.config(text=f"Predicted Digit: {predicted_class}")
        print(f"Prediction: {predicted_class}, Raw Vector: {prediction_vector}")

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

if __name__ == '__main__':
    root = tk.Tk()
    gui = DigitRecognizerGUI(root)
    root.mainloop()
