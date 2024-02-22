import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.model_name = 'Linear Regression'
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coeff = None
        self.intercept = 0

    @staticmethod
    def fit_input(X):
        if isinstance(X, np.ndarray):
            print('hello np')
            return X
        elif isinstance(X, pd.DataFrame):
            print('hello pd')
            return X.values
        else:
            raise TypeError('Unsupported input type')

    def fit(self, X, y):
        print(X.shape, 'f')

        X = self.fit_input(X)
        print(X.shape, 'f')

        m, n = X.shape
        self.coeff = np.zeros(n)
        self.intercept = 0
        print(X.shape, 'f')
        return self._gradient_descent(X, y)

    def compute_cost_function(self, X, y):
        m, n = X.shape
        y_pred = np.dot(X, self.coeff) + self.intercept
        y_pred = y_pred.reshape(-1, 1)

        sq_diff = np.square(y_pred - y)
        print(sq_diff.shape, 'llllllllllllllllllllllllllll')
        total_cost = np.sum(sq_diff) / (2 * m)
        return total_cost

    def _compute_gradients(self, X, y):
        # for each feature create a gradient
        # for each feature create a gradient
        m, n = X.shape

        y_pred = np.dot(X, self.coeff) + self.intercept
        y_pred = y_pred.reshape(-1, 1)

        dj_dw = np.zeros(n)
        dj_db = 0
        print(X.shape, ';')
        for j in range(n):
            for i in range(m):
                dj_dw[j] += ((y_pred[i][0] - y[i][0]) * X[i, j]) / m

            print(dj_dw, 'kk')

        for i in range(m):
            dj_db += (y_pred[i][0] - y[i][0]) / m

        print(dj_db, 'hhh')
        return dj_dw, dj_db


    def _gradient_descent(self, X, y):
        m, n = X.shape
        cost = []
        for i in range(self.num_iterations):
            print(f'gradient descent {i + 1}/{self.num_iterations}')
            print(self.coeff, self.intercept)
            print(X.shape, 'g')
            dj_dw, dj_db = self._compute_gradients(X, y)
            for j in range(n):
                print(dj_dw[j])
                self.coeff[j] -= self.learning_rate * dj_dw[j]

            self.intercept -= self.learning_rate * dj_db

            cost.append(self.compute_cost_function(X, y))
        print('gradient descent has finished.')
        return cost

    def predict(self, X):
        X = self.fit_input(X)
        y_pred = np.dot(X, self.coeff) + self.intercept
        return y_pred.reshape(-1, 1)





