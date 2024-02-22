import numpy as np
from matplotlib import pyplot as plt
from mysklearn.linear_regression import *
def multi_variate():
    # multi_variate()
    # data - don't forget to normalize data if needed
    X = np.linspace(0, 10, num=1000).reshape(-1, 1)
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
    print(X.shape)
    y = X_norm * 0.5 + 1

    # visualize X and y
    # plt.scatter(X_norm, y)
    # plt.show()
    df = pd.read_csv('../datasets/ex1data1.csv')
    print(df.head)
    df.columns = ['column1', 'column2']

    X_norm1 = df['column1'].values
    X_norm2 = (df['column1'].values * -0.2 - 11)
    print(X_norm2.shape)
    X_norm = df.iloc[:, :-1].values

    print(X_norm.shape)
    y = df.iloc[:, -1].values.reshape(-1, 1)
    print("x, y shapes:",X_norm.shape, y.shape)
    # return
    # linear_regression model
    reg = LinearRegression(num_iterations=1500)
    cost = reg.fit(X_norm, y)
    y_pred = reg.predict(X_norm)
    print(f"coeff : {reg.coeff}\nintercept : {reg.intercept}")
    print('cost : ', cost[-1])

    # plot cost during training
    x_axis = np.linspace(0, 10, num=1500)  # nums = num_iterations
    plt.plot(x_axis, cost)
    plt.show()
    X = X_norm
    # plot predictions
    plt.scatter(X[:, 0], y)
    plt.plot(X[:,0], y_pred, color='red')
    plt.show()

if __name__ == '__main__':
    multi_variate()
    # # data - don't forget to normalize data if needed
    # X = np.linspace(0, 10, num=1000).reshape(-1, 1)
    # X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
    # print(X.shape)
    # y = X_norm * 0.5 + 1
    #
    # # visualize X and y
    # # plt.scatter(X_norm, y)
    # # plt.show()
    # df = pd.read_csv('../datasets/ex1data1.csv')
    # print(df.head)
    # df.columns = ['column1', 'column2']
    #
    # X_norm = df['column1'].values.reshape(-1, 1)
    # y = df['column2'].values.reshape(-1, 1)
    # print(X.shape, y.shape)
    # # linear_regression model
    # reg = LinearRegression(num_iterations=1500)
    # cost = reg.fit(X_norm, y)
    # y_pred = reg.predict(X_norm)
    # print(f"coeff : {reg.coeff}\nintercept : {reg.intercept}")
    # print('cost : ', cost[-1])
    #
    # # plot cost during training
    # x_axis = np.linspace(0, 10, num=1500)  # nums = num_iterations
    # plt.plot(x_axis, cost)
    # plt.show()
    # X = X_norm
    # # plot predictions
    # plt.scatter(X[:, 0], y)
    # plt.plot(X[:,0], y_pred, color='red')
    # plt.show()
