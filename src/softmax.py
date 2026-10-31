"""Multiclass softmax (multinomial logistic) regression trained with full-batch gradient descent."""
import numpy as np


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # subtract row max for numerical stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, n_classes=10, lr=0.5, l2=1e-4, epochs=3000, tol=1e-7):
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.tol = tol

    def loss(self, X, y):
        """Mean cross-entropy plus (l2 / 2) * ||W||^2 (bias is not regularized)."""
        P = self.predict_proba(X)
        nll = -np.log(P[np.arange(len(y)), y] + 1e-12).mean()
        return nll + 0.5 * self.l2 * np.sum(self.W ** 2)

    def gradients(self, X, y):
        n = len(y)
        P = self.predict_proba(X)
        P[np.arange(n), y] -= 1.0  # P - Y, where Y is one-hot
        grad_W = X.T @ P / n + self.l2 * self.W
        grad_b = P.mean(axis=0)
        return grad_W, grad_b

    def fit(self, X, y, X_val=None, y_val=None):
        self.W = np.zeros((X.shape[1], self.n_classes))
        self.b = np.zeros(self.n_classes)
        self.history = {"train_loss": [], "val_loss": []}
        prev = np.inf
        for _ in range(self.epochs):
            grad_W, grad_b = self.gradients(X, y)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
            cur = self.loss(X, y)
            self.history["train_loss"].append(cur)
            if X_val is not None:
                self.history["val_loss"].append(self.loss(X_val, y_val))
            if abs(prev - cur) < self.tol:
                break
            prev = cur
        self.n_epochs_run = len(self.history["train_loss"])
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    @property
    def n_params(self):
        return self.W.size + self.b.size
