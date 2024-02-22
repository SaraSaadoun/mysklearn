import numpy as np
import matplotlib.pyplot as plt
from my_sklearn.perceptron import Perceptron


if __name__ == '__main__':
    # dataset
    features = np.array([[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 2]])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # perceptron
    p = Perceptron(num_iterations=150)
    weights, bias, errors = p.fit(features, labels)
    print(weights, bias, errors[-1])

    # visualize result
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8,8), gridspec_kw={'height_ratios': [2, 1]})
    ax1.scatter(features[:, 0], features[:, 1], c=labels)
    x = np.linspace(0, 3, 30)
    ax1.plot(x, (-weights[0] * x - bias) / weights[1])

    # visualize convergence
    ax2.plot(range(len(errors)), errors)
    plt.show()
