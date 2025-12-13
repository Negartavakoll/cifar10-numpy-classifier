import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr

        # Weight initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)

        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_backward(self, dz, z):
        return dz * (z > 0)

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = self.softmax(self.z2)

        return self.out

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(np.log(y_pred[range(m), y_true])) / m

    def backward(self, X, y_true):
        m = X.shape[0]

        dz2 = self.out
        dz2[range(m), y_true] -= 1
        dz2 /= m

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = self.relu_backward(da1, self.z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
