import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.5, epochs=1000, l2=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained.")
        z = np.asarray(X @ self.weights).reshape(-1) + self.bias
        return self.sigmoid(z)

    def _loss(self, y, p):
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        data_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        penalty = (self.l2 / (2 * len(y))) * np.sum(self.weights ** 2)
        return float(data_loss + penalty)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        n, features = X.shape
        if n != len(y):
            raise ValueError("X and y contain different sample counts.")

        self.weights = np.zeros(features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            p = self.predict_proba(X)
            error = p - y
            dw = np.asarray(X.T @ error).reshape(-1) / n
            dw += (self.l2 / n) * self.weights
            db = float(np.mean(error))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch == 0 or (epoch + 1) % 10 == 0:
                self.loss_history.append(self._loss(y, self.predict_proba(X)))
        return self

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
