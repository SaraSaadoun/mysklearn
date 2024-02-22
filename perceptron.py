import numpy as np


class Perceptron :
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coeff = None
        self.intercept = 0
        self.threshold_limit = 0

    def score(self, x):
        return np.dot(x, self.coeff) + self.intercept

    def step(self, y_score):
        return y_score >= 0

    def prediction(self, x):
        return self.step(self.score(x))

    def error(self, x, y):
        y_pred = self.prediction(x)
        if y_pred == y:
            return 0
        return np.abs(self.score(x))

    def mean_perceptron_error(self, X, y):
        total_error = 0
        m, n = X.shape
        for i in range(m):
            total_error += np.mean(self.error(X[i], y[i]))
        return total_error / m

    def update_step(self, x, y):
        y_pred = self.prediction(x)

        updated_coeff = self.coeff + self.learning_rate * (y - y_pred) * x
        updated_intercept = self.intercept + self.learning_rate * (y - y_pred)

        return updated_coeff, updated_intercept

    def fit(self, X, y):
        m, n = X.shape
        errors = []

        # init
        self.coeff = np.random.rand(n)
        self.intercept = 0.0

        for _ in range(self.num_iterations):
            errors.append(self.mean_perceptron_error(X, y))
            idx = np.random.randint(m)
            self.coeff, self.intercept = self.update_step(X[idx], y[idx])

        return self.coeff, self.intercept, errors


