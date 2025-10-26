# GUI Predictor Documentation

This document details the `gui_predictor.py` script, which provides an interactive graphical user interface (GUI) for real-time handwritten digit recognition using our trained Convolutional Neural Network.

## Purpose

The `gui_predictor.py` script serves as a practical demonstration of the CNN's capabilities. It allows users to draw digits directly on a canvas and instantly see the model's prediction along with confidence scores for each possible digit class. This interactive feedback loop is invaluable for understanding how the model interprets various drawing styles and for debugging potential issues.

## Core Components and Functionality

### 1. Drawing Mechanism

*   **Canvas:** The GUI presents a `tkinter.Canvas` element that visually represents a 28x28 pixel grid, matching the input dimensions of our CNN.
*   **Brush Stroke:** When the user draws on the canvas, individual pixels are activated. The current implementation uses a **1-pixel brush stroke**, meaning only the exact pixel under the mouse cursor is marked. This provides precise control over the input digit.
*   **`image_array`:** The state of the drawn digit is internally maintained in a NumPy array (`self.image_array`), which is a 28x28 matrix where `0` represents a black (empty) pixel and `1` represents a white (drawn) pixel.

### 2. Image Preprocessing (`process_image_for_prediction` method)

This method is crucial for the model's robust performance, especially when digits are drawn off-center or with varying sizes. It mimics the preprocessing steps applied to the MNIST training data.

*   **Functionality:**
    1.  **Bounding Box Detection:** It first identifies the tightest bounding box around the drawn digit by finding the minimum and maximum active rows and columns in `self.image_array`.
    2.  **Cropping:** The digit is then cropped to this bounding box, removing unnecessary empty space.
    3.  **Aspect Ratio Correction (Padding to Square):** The cropped digit is padded with zeros to make it perfectly square while preserving its aspect ratio. This is vital because MNIST digits are typically square, and maintaining this aspect ratio prevents distortion.
    4.  **Resizing:** The square digit is then resized to 20x20 pixels using high-quality anti-aliasing (LANCZOS filter). This downsampling helps in standardizing the input size.
    5.  **Centering:** Finally, the 20x20 digit is padded with 4 pixels of zeros on all sides, effectively centering it within a 28x28 canvas. This ensures that the model always receives a consistently sized and centered input, regardless of where the user drew the digit on the larger canvas.
    6.  **Normalization:** The pixel values are normalized to the range [0, 1] and reshaped to `(1, 28, 28, 1)` to match the CNN's expected input format.

*   **Why it's Important:** This preprocessing pipeline significantly enhances the model's ability to generalize to diverse user drawings, even if they are not perfectly centered or sized on the initial canvas. It ensures that the model receives a standardized representation of the digit, similar to what it was trained on.

### 3. Real-time Prediction and Confidence Display

*   **Trigger:** The `predict_realtime` method is called continuously as the user draws (on mouse motion).
*   **Prediction:** The preprocessed image is fed into the loaded CNN (`self.net.predict()`), which outputs a probability distribution over the 10 digit classes.
*   **Confidence Bars:** The GUI dynamically updates a set of `ttk.Progressbar` widgets, where each bar represents a digit (0-9) and its length corresponds to the model's predicted confidence (probability) for that digit. The predicted class is highlighted.
*   **Prediction Label:** A prominent label displays the digit with the highest predicted probability.

## Usage

To run the GUI, ensure you have a trained model (e.g., `models/mnist_cnn_full_dataset.npz`) and execute:

```bash
python gui_predictor.py
```

This will launch the interactive digit recognition application.