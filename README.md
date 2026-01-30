# CIFAR-10 Image Classification from Scratch (NumPy)

This project implements an image classification pipeline for the CIFAR-10 dataset **from scratch using NumPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch.

The goal of this project is to demonstrate a clear understanding of:
- Data preprocessing
- Neural network fundamentals
- Forward and backward propagation
- Training and evaluation logic
- Machine learning implementation without high-level libraries

---

## 📌 Project Overview

- Dataset: CIFAR-10 (32×32 RGB images, 10 classes)
- Model: Fully connected Neural Network
- Implementation: Pure NumPy
- Task: Multiclass image classification

---

## 📂 Project Structure
# CIFAR-10 Image Classification from Scratch (NumPy)

This project implements an image classification pipeline for the CIFAR-10 dataset **from scratch using NumPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch.

The goal of this project is to demonstrate a clear understanding of:
- Data preprocessing
- Neural network fundamentals
- Forward and backward propagation
- Training and evaluation logic
- Machine learning implementation without high-level libraries

---

## 📌 Project Overview

- Dataset: CIFAR-10 (32×32 RGB images, 10 classes)
- Model: Fully connected Neural Network
- Implementation: Pure NumPy
- Task: Multiclass image classification

---

## 📂 Project Structure
cifar10-numpy-classifier/
│
├── data/
│ └── cifar10_small/
│ ├── train_images.npy
│ ├── train_labels.npy
│ ├── test_images.npy
│ └── test_labels.npy
│
├── src/
│ ├── train.py # Training loop
│ ├── model.py # Neural network implementation
│ ├── utils.py # Data loading & preprocessing
│ └── metrics.py # Accuracy calculation
│
├── README.md
└── requirements.txt
🧠 Model Architecture

- Input layer: 3072 neurons (32 × 32 × 3)
- Hidden layer(s): Fully connected
- Output layer: 10 neurons (softmax)
- Loss function: Cross-entropy loss
- Optimization: Gradient Descent

---

## ⚙️ Data Preprocessing

- Images are reshaped from `(32, 32, 3)` to `(3072,)`
- Pixel values are normalized to the range `[0, 1]`
- Labels are integer-encoded (0–9)

🛠 Technologies Used
Python
NumPy
Git & GitHub

🎯 Learning Outcomes
Implemented a neural network without ML frameworks
Understood backpropagation at a low level
Practiced project structuring for ML workflows
Gained experience preparing projects for GitHub and portfolios

🎯 Learning Outcomes
Implemented a neural network without ML frameworks
Understood backpropagation at a low level
Practiced project structuring for ML workflows
Gained experience preparing projects for GitHub and portfolios

👤 Author
Negar Tavakol
Computer Engineering – Software
Machine Learning & Applied AI


      
