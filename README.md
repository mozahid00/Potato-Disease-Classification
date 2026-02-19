# Potato Disease Classification using CNN

Project Overview

This project builds a Convolutional Neural Network (CNN) model to classify potato leaf images into three categories:

Potato Early Blight

Potato Late Blight

Healthy Potato Leaf

The model is trained on the PlantVillage dataset and implemented using TensorFlow/Keras in Google Colab.

# Dataset

Dataset used: PlantVillage (Potato subset)

Source: Kaggle (https://www.kaggle.com/datasets/arjuntejaswi/plant-village/data)

The dataset contains labeled images organized in folders:

Potato___Early_blight

Potato___Late_blight

Potato___healthy


Total classes: 3

Image type: RGB

Input size: 224 × 224

# Tools Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Google Colab

# Model Architecture

The CNN model consists of:

Conv2D (32 filters)

MaxPooling

Conv2D (64 filters)

MaxPooling

Conv2D (128 filters)

MaxPooling

Flatten

Dense (128 units)

Dropout

Output layer (Softmax)

Loss Function: Categorical Crossentropy
Optimizer: Adam
Metric: Accuracy

# Data Preprocessing

Image resizing to 224×224

Pixel normalization (1./255)

Data augmentation:

Rotation

Horizontal flip

Zoom

Train/Validation split: 80/20

# Results

Final Training Accuracy: 98%
Final Validation Accuracy: 94%

Include screenshots of:
<img width="922" height="201" alt="image" src="https://github.com/user-attachments/assets/bd2d28db-6b29-4990-8ea9-daf6ee9a267f" />

Accuracy vs Epoch plot & Loss vs Epoch plot
<img width="1033" height="474" alt="image" src="https://github.com/user-attachments/assets/2803228e-878d-4fc6-818f-de81ef244b44" />

Sample predictions
<img width="1161" height="408" alt="image" src="https://github.com/user-attachments/assets/53a52f4f-09a8-4b59-b42c-06bfc674b672" />

# How to Run This Project

Clone this repository

Open the notebook in Google Colab

Upload the dataset or connect Google Drive

Run all cells

Train the model

Evaluate results

# Model Saving

The trained model is saved as:

potato_disease_classifier.h5

You can load it using:

from tensorflow.keras.models import load_model
model = load_model("potato_disease_model.h5")

# Future Improvements

Use Transfer Learning (MobileNet / ResNet)

Add Confusion Matrix

Deploy using Streamlit

Improve generalization with more augmentation

-Author-

S M Mozahidul Haque

LinkedIn: www.linkedin.com/in/mozahidul-haque

GitHub: https://github.com/mozahid00
