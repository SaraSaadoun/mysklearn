import numpy as np
from matplotlib import pyplot as plt
from mysklearn.linear_regression import *


def univariate_example():
    # data - don't forget to normalize data if needed
    X = np.linspace(0, 10, num=1000).reshape(-1, 1)
    y = X * 0.5 + 1
    # normalizing data
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))

    print("x, y shapes:", X_norm.shape, y.shape)

    # linear_regression model
    num_iterations = 1500
    reg = LinearRegression(num_iterations=num_iterations)
    cost = reg.fit(X_norm, y)
    y_pred = reg.predict(X_norm)
    print(f"coeff : {reg.coeff[0]}\nintercept : {reg.intercept}")
    print('cost : ', cost[-1])

    # plot cost during training
    x_axis = np.linspace(0, 10, num=num_iterations)
    plt.plot(x_axis, cost)
    plt.title("cost during training")
    plt.show()

    # plot predictions
    plt.scatter(X_norm[:, 0], y)
    plt.plot(X_norm[:, 0], y_pred, color='red')
    plt.title("feature with target")
    plt.show()


def multi_variate_example():
    # loading data
    df = pd.read_csv('../datasets/ex1data1.csv')
    print(df.head)

    X = (df.iloc[:, :].values)
    y = (0.3 * X[:, 0] - 0.7 * X[:, 1] + 0.9)

    # normalizing data
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))

    print("x, y shapes:", X_norm.shape, y.shape)

    # linear_regression model
    num_iterations = 1500
    reg = LinearRegression(num_iterations=num_iterations)
    cost = reg.fit(X_norm, y)
    y_pred = reg.predict(X_norm)
    print(f"coeff : {reg.coeff[0]}\nintercept : {reg.intercept}")
    print('cost : ', cost[-1])

    # plot cost during training
    x_axis = np.linspace(0, 10, num=num_iterations)  #
    plt.plot(x_axis, cost)
    plt.title("cost during training")
    plt.show()

    # plot predictions
    plt.scatter(X_norm[:, 0], y)
    plt.plot(X_norm[:, 0], y_pred, color='red')
    plt.title("first feature with target")
    plt.show()

    plt.scatter(X_norm[:, 1], y)
    plt.plot(X_norm[:, 1], y_pred, color='red')
    plt.title("second feature with target")
    plt.show()


if __name__ == '__main__':
    # try univariate or multivariate
    univariate_example()
    # multi_variate_example()

