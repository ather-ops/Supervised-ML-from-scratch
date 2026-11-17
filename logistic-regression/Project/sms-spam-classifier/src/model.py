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

    def _loss(self, y, p, sample_weights):
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        individual_loss = -(
            y * np.log(p) +
            (1 - y) * np.log(1 - p)
        )

        data_loss = np.mean(sample_weights * individual_loss)
        penalty = (
            self.l2 / (2 * len(y))
        ) * np.sum(self.weights ** 2)
        return float(data_loss + penalty)

    def fit(self, X, y):

        y = np.asarray(y, dtype=float).reshape(-1)
        n, features = X.shape
        if n != len(y):
            raise ValueError("X and y contain different sample counts.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Labels must contain only 0 and 1.")

        self.weights = np.zeros(features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)

        if n_negative == 0 or n_positive == 0:
            raise ValueError("Training data must contain both classes.")

        weight_negative = n / (2 * n_negative)
        weight_positive = n / (2 * n_positive)

        sample_weights = np.where(
            y == 1,
            weight_positive,
            weight_negative
        )

        print("\nClass-weighted training active!")
        print("HAM weight:", weight_negative)
        print("SPAM weight:", weight_positive)


        for epoch in range(self.epochs):

            p = self.predict_proba(X)
            error = (p - y) * sample_weights
            dw = np.asarray(X.T @ error).reshape(-1) / n
            dw += (self.l2 / n) * self.weights
            db = float(np.mean(error))
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch == 0 or (epoch + 1) % 10 == 0:

                current_loss = self._loss(
                    y,
                    self.predict_proba(X),
                    sample_weights
                )

                self.loss_history.append(current_loss)
        return self

    def predict(self, X, threshold=0.5):
        return (
            self.predict_proba(X) >= threshold
        ).astype(int)
