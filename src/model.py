import numpy as np

class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = 0.001 * np.random.randn(input_dim, num_classes)
        self.b = np.zeros((1, num_classes))

    def forward(self, X):
        logits = X @ self.W + self.b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def train(self, X, y_onehot, lr=0.01):
        N = X.shape[0]
        probs = self.forward(X)

        # loss not printed
        grad_logits = (probs - y_onehot) / N
        self.W -= lr * (X.T @ grad_logits)
        self.b -= lr * np.sum(grad_logits, axis=0, keepdims=True)
