import numpy as np
from utils import load_image_folder, one_hot_encode, accuracy
from model import SoftmaxClassifier
from model_mlp import MLP


print("Loading training data...")
X_train, y_train, class_names = load_image_folder("C:/Users/ghtav/PycharmProjects/cifar10-numpy-classifier/src/data/cifar10_synthetic_small/train")
X_test, y_test, _ = load_image_folder("C:/Users/ghtav/PycharmProjects/cifar10-numpy-classifier/src/data/cifar10_synthetic_small/test")


# Flatten images from (32,32,3) → (3072,)
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean


print("Checking image shape...")
print(X_train[0].reshape(32, 32, 3).shape)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

num_classes = len(class_names)
print("Number of classes:", num_classes)

# One-hot encoding labels
y_train_oh = one_hot_encode(y_train, num_classes)
y_test_oh = one_hot_encode(y_test, num_classes)

model = SoftmaxClassifier(input_dim=3072, num_classes=num_classes)

model = MLP(
    input_dim=3072,
    hidden_dim=128,
    output_dim=num_classes,
    lr=0.1

)



epochs = 50

batch_size = 64

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        preds = model.forward(X_batch)
        loss = model.compute_loss(y_batch, preds)
        model.backward(X_batch, y_batch)

    test_preds = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test)

    print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_acc:.3f}")

print("Training complete!")
