import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.model_name = 'Linear Regression'
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coeff = None
        self.intercept = 0
        self.n = 0
        self.m = 0
        self.X = None
        self.y = None

    @staticmethod
    def fit_input(X):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, pd.DataFrame):
            return X.values
        else:
            raise TypeError('Unsupported input type')

    def fit(self, X, y):
        # fix input shapes and types as 2d np arrays
        self.X = self.fit_input(X)
        self.y = y.reshape(-1, 1)

        self.m, self.n = self.X.shape

        return self._gradient_descent()


    def compute_cost_function(self):
        # implementation of mean square error (MSE)
        y_pred = self.predict(self.X)

        sq_diff = np.square(y_pred - self.y)
        total_cost = np.sum(sq_diff) / (2 * self.m)
        return total_cost

    def _compute_gradients(self):
        y_pred = self.predict(self.X)
        dj_dw = (2 * np.dot(self.X.T, (y_pred - self.y))) / self.m
        dj_db = 2 * np.sum(y_pred - self.y) / self.m

        # iterative implementation (CHECKING)
        # dw = np.zeros((self.n, 1))
        # for j in range(self.n):
        #     for i in range(self.m):
        #         dw[j] += 2 * (y_pred[i][0] - self.y[i][0]) * self.X[i, j]
        # dw /= self.m
        # assert (dw.all() == dj_dw.all()), f'Values didn\'t match. Found: {dw}  Expected: {dj_dw}'
        #
        # db = 0
        # for i in range(self.m):
        #     db += 2 * (y_pred[i][0] - self.y[i][0])
        # db /= self.m
        # assert (db.all() == dj_db.all()), f'Values didn\'t match. Found: {db}  Expected: {dj_db}'

        return dj_dw, dj_db


    def _gradient_descent(self):
        # initialization
        cost = []
        self.coeff = np.zeros((self.n, 1))
        self.intercept = 0
        for i in range(self.num_iterations):
            print(f'epoch {i + 1}/{self.num_iterations}')
            # compute gradients
            dj_dw, dj_db = self._compute_gradients()
            # update gradients
            self.coeff -= self.learning_rate * dj_dw
            self.intercept -= self.learning_rate * dj_db

            cost.append(self.compute_cost_function())
        print('training has finished.')
        return cost

    def predict(self, X):
        X = self.fit_input(X)
        y_pred = np.dot(X, self.coeff) + self.intercept
        return y_pred.reshape(-1, 1)





