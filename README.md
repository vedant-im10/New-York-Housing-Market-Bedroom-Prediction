# 🏠 New York Housing Market - Bedroom Prediction

This project implements a neural network model to predict the number of bedrooms in properties listed on the New York housing market. The model was built from scratch, using Numpy as the only external library, and achieves a classification accuracy based on various property features.

## 💡 Project Overview

The task involves predicting the number of bedrooms based on 16 other features, such as price, property square footage, and location attributes. The dataset consists of 4801 records, each with multiple attributes related to real estate sales.

## 📋 Key Features

- **Neural Network Architecture**: A multi-layer perceptron (MLP) with two hidden layers, each containing 64 nodes.
- **Preprocessing**: One-hot encoding for categorical features and normalization for continuous features.
- **Optimization**: The model uses the Adam optimizer and tunes various hyperparameters such as learning rate and number of epochs.
- **Evaluation**: The model was evaluated based on accuracy across different test/train splits.

## 🛠 Methodology

1. **Data Preprocessing**: 
   - Applied one-hot encoding to categorical data (e.g., house type).
   - Normalized continuous features like price, latitude, and longitude.

2. **Model Architecture**: 
   - Input layer with 16 features.
   - Two hidden layers with 64 nodes each, using the ReLU activation function.
   - Output layer using softmax for multi-class classification.

3. **Hyperparameter Tuning**: 
   - Experimented with different learning rates and the number of nodes per layer.
   - Trained over 400 epochs using the Adam optimizer.

4. **Training and Evaluation**:
   - Used 5 different data splits to evaluate performance, achieving an average accuracy of ~98%.

## 🔍 Results

- **Training Logs**: The cost decreased significantly over iterations, indicating proper model learning.
- **Test Accuracy**: The model achieved approximately 98% accuracy in predicting the correct number of bedrooms.
