# Image Classification with Neural Network (NumPy)

This project implements a simple multi-layer neural network **from scratch using NumPy** to perform image classification on a CIFAR-10–like dataset.

## Project Overview
- Task: Image classification (10 classes)
- Dataset: CIFAR-10–style images (32×32 RGB)
- Model: Fully connected neural network (MLP)
- Frameworks: **Pure NumPy (no PyTorch / TensorFlow)**

## Model Architecture
- Input layer: 32×32×3 (flattened)
- Hidden layer: Fully connected + ReLU
- Output layer: Softmax
- Loss: Categorical Cross-Entropy
- Optimization: Mini-batch Gradient Descent

## Training Details
- Weight initialization: He initialization
- Learning rate: Tuned manually
- Batch size: 64
- Evaluation metric: Accuracy

## Results
- Achieved ~18% test accuracy on a small-scale CIFAR-10–like dataset
- Demonstrates correct forward/backward propagation and learning behavior

> Note: Performance is intentionally limited due to the use of a simple MLP and a small dataset.  
> The goal of this project is educational: understanding neural networks at a low level.

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
