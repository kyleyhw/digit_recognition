import tkinter as tk
from tkinter import ttk, Canvas
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.network import Network
from src.layers import Convolutional, ReLU, MaxPooling, Flatten, Dense, Softmax


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.rows = 28
        self.cols = 28
        self.pixel_size = 15  # Increased pixel size for a larger canvas
        self.canvas_width = self.cols * self.pixel_size
        self.canvas_height = self.rows * self.pixel_size

        # Main frame
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        # Canvas for drawing
        self.canvas = Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=(0, 15))

        # --- Confidence meters ---
        confidence_frame = tk.Frame(main_frame)
        confidence_frame.pack(side=tk.RIGHT)

        self.confidence_labels = []
        self.confidence_bars = []
        for i in range(10):
            label_frame = tk.Frame(confidence_frame)
            label_frame.pack(fill="x", pady=2)

            label = tk.Label(label_frame, text=f"{i}:", font=("Helvetica", 12), width=3)
            label.pack(side=tk.LEFT)

            bar = ttk.Progressbar(label_frame, length=150, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)

            percent_label = tk.Label(label_frame, text="0.00%", font=("Helvetica", 12), width=7, anchor="w")
            percent_label.pack(side=tk.LEFT)

            self.confidence_labels.append((label, percent_label))
            self.confidence_bars.append(bar)

        # --- Prediction and Clear Button ---
        self.prediction_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 24, "bold"))
        self.prediction_label.pack(pady=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas, font=("Helvetica", 14))
        self.clear_button.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image_array = np.zeros((self.rows, self.cols))
        self.create_grid()

        self.load_network()

    def create_grid(self):
        self.pixel_grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.pixel_size
                y1 = r * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.pixel_grid[r][c] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="#222")

    def paint(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        # Simple anti-aliasing by using a brush
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                if 0 <= c < self.cols and 0 <= r < self.rows:
                    # Calculate distance for intensity falloff
                    dist = np.sqrt((r - row) ** 2 + (c - col) ** 2)
                    intensity = max(0, 1.0 - dist / 1.5)  # Adjust falloff radius
                    new_val = self.image_array[r, c] + intensity
                    self.image_array[r, c] = min(1.0, new_val)

                    # Update color based on intensity
                    gray_val = int(self.image_array[r, c] * 255)
                    self.canvas.itemconfig(self.pixel_grid[r][c], fill=f'#%02x%02x%02x' % (gray_val, gray_val, gray_val))
        self.predict_realtime()

    def clear_canvas(self):
        self.image_array = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                self.canvas.itemconfig(self.pixel_grid[r][c], fill="black")
        self.prediction_label.config(text="Prediction: ")
        for i in range(10):
            self.confidence_labels[i][0].config(bg="SystemButtonFace")
            self.confidence_labels[i][1].config(text="0.00%")
            self.confidence_bars[i]['value'] = 0

    def process_image_for_prediction(self):
        # Find bounding box
        rows = np.any(self.image_array, axis=1)
        cols = np.any(self.image_array, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to bounding box
        digit_image = self.image_array[rmin:rmax + 1, cmin:cmax + 1]

        # Get dimensions and calculate padding
        h, w = digit_image.shape
        delta = abs(h - w)
        pad_top, pad_bottom = (delta // 2, delta - (delta // 2)) if h < w else (0, 0)
        pad_left, pad_right = (delta // 2, delta - (delta // 2)) if w < h else (0, 0)

        # Pad to square
        digit_padded = np.pad(digit_image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

        # Resize to 20x20 with anti-aliasing and place in 28x28 frame
        img = Image.fromarray((digit_padded * 255).astype(np.uint8))
        img = img.resize((20, 20), Image.LANCZOS)
        img_padded = np.pad(np.array(img), ((4, 4), (4, 4)), 'constant', constant_values=0)

        # Normalize and reshape for the network
        img_final = img_padded.astype('float32') / 255.0
        return img_final.reshape(1, 28, 28, 1)

    def predict_realtime(self):
        processed_image = self.process_image_for_prediction()
        if processed_image is None:
            return  # Nothing to predict

        prediction_vector = self.net.predict(processed_image)
        probabilities = prediction_vector.flatten()
        prediction = np.argmax(probabilities)

        for i in range(10):
            confidence = probabilities[i] * 100
            self.confidence_labels[i][1].config(text=f"{confidence:.2f}%")
            self.confidence_bars[i]['value'] = confidence
            if i == prediction:
                self.confidence_labels[i][0].config(bg="yellow")
            else:
                self.confidence_labels[i][0].config(bg="SystemButtonFace")

        self.prediction_label.config(text=f"Prediction: {prediction}")

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

        model_path = "models/mnist_cnn_subset_1000.npz"
        try:
            print(f"\n--- Loading the Trained Model from '{model_path}' ---")
            self.net.load_model(model_path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"\nERROR: Model file '{model_path}' not found.")
            print("Please run 'python train_mnist.py' to train and save the model.")
            self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()