# HANDWRITTEN_DIGITS_IDENTIFIER
Handwritten Digit Classification with Convolutional Neural Networks (CNN)
An end-to-end tutorial and implementation notebook demonstrating how to build, train, and test a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9).

This project covers the full pipeline: training on the standard MNIST dataset, preprocessing custom user images using OpenCV and Pillow (PIL), and feeding them into the trained model for inference.

🚀 Features
Part 1: CNN Training on MNIST

Loads and normalizes the 70,000-image MNIST dataset.

Constructs a robust Convolutional Neural Network using tensorflow.keras.

Evaluates model performance on unseen test data.

Part 2: Custom Image Prediction using OpenCV

Resizes and converts custom phone or scanner photos of handwritten digits into the exact format expected by the model.

Applies thresholding and resizing techniques to mimic MNIST's white-on-black data distribution.

Part 3: Custom Image Prediction using Pillow (PIL)

Provides an alternative image preprocessing pipeline utilizing Python's Pillow library for environments where OpenCV is not available.

🛠️ Tech Stack
Python 3.x

TensorFlow / Keras (Deep learning framework)

NumPy (Numerical matrix operations)

Matplotlib (Data visualization and plotting)

OpenCV & Pillow (Image processing and manipulation)

📦 Getting Started
Prerequisites
Make sure you have the required Python libraries installed:

Bash
pip install tensorflow numpy matplotlib opencv-python pillow
Running the Notebook
Clone the repository or download the handwritten_digits_cnn.ipynb file.

Launch Jupyter Notebook or JupyterLab:

Bash
jupyter notebook
Open handwritten_digits_cnn.ipynb and execute the cells sequentially from top to bottom.

📝 Detailed Project Description
Overview
Optical Character Recognition (OCR) for handwritten digits is a foundational challenge in computer vision. While datasets like MNIST make it straightforward to train models under controlled conditions, bridging the gap between clean dataset images and real-world inputs (such as photos taken with a smartphone) introduces significant data distribution challenges.

This project explores how a Convolutional Neural Network can be trained to recognize features like loops, lines, and intersections in handwritten numbers, and investigates the necessary image preprocessing pipelines required to feed custom, real-world data into a trained model successfully.

Architectural Design & Pipeline
1. Model Architecture
The neural network is built as a sequential stack of layers designed to extract spatial hierarchies from image data:

Input Layer: Accepts 28×28 grayscale pixel grids.

Convolutional Layers (Conv2D): Apply trainable filters to detect low-level features (edges, curves) and high-level structural patterns.

Max Pooling Layers (MaxPooling2D): Downsample the feature maps to reduce spatial dimensions and computational complexity while retaining dominant signals.

Dropout Layer: Randomly deactivates a fraction of neurons during training to mitigate overfitting.

Dense Layers: Fully connected layers that map the extracted feature representations to final classification probabilities across 10 classes (digits 0 through 9 using a softmax activation function).

2. The Preprocessing Challenge
A model trained on MNIST expects inputs adhering to specific properties:

Background vs. Foreground: White digits on a black background (
pixel value∈[0,255]).

Centering and Scale: The digit is tightly cropped and centered within a 28×28 pixel canvas.

Real-world photos typically feature dark ink on light paper, ambient lighting gradients, and unpredictable bounding boxes. To solve this, the notebook incorporates robust image manipulation steps using both OpenCV and Pillow:

Grayscale Conversion: Strips out unnecessary color channels.

Inversion & Thresholding: Converts dark-on-light inputs to light-on-dark representations matching MNIST training standards.

Resizing & Normalization: Rescales pixel matrices to fit the 
28×28 dimension and normalizes values to a float range between 0.0 and 1.0.

Results & Takeaways
The CNN achieves high classification accuracy on the MNIST test partition.

Preprocessing alignment dictates real-world success: minor shifts in thresholding or stroke thickness can significantly alter prediction confidence, highlighting the importance of robust data preprocessing pipelines when deploying computer vision models in production.
