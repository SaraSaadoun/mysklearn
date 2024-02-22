import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mysklearn.k_means import KMeans

# helper functions
def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # normalize
    return img


def show_img(img):
    plt.imshow(img)
    plt.show()

# examples
def image_compression():
    # init KMeans instance with 16 cluster and iteration number is 10
    k_means = KMeans(n_clusters=16, n_iterations=10)

    # read and show the image needed to be compressed
    img = read_image('../datasets/bird.JPG')
    show_img(img)

    # reshape it from (n, m, 3 ) in which the image is n * m pixels and 3 refer to 3 rgb colors to (n * m, 3)
    img_2d = img.reshape((-1, img.shape[2]))

    # fit the model - get centroids values
    # centroids values refer to the color palette which consists of 16 color (clusters)
    k_means.fit(img_2d)

    # get an array of each pixel (index) mapped to nearest color of the 16 color
    idx = k_means.predict(img_2d)

    # repaint the image with the new palette :)
    new_img = k_means.centroids[idx, :]

    # restore its shape to be viewed
    new_img = np.reshape(new_img, img.shape)

    # wish you a happy result :)
    show_img(new_img)


def housing():
    # read dataset
    df = pd.read_csv('../datasets/housing.csv', header=0)

    # extract only these 3 features
    X = df.loc[:, ['Latitude', 'Longitude', 'MedInc']]
    print(X.head())
    # print(X.isna().sum())

    # init an instance of KMeans with 4 clusters
    k_means = KMeans(n_clusters=4, n_iterations=50)
    # fit the model
    k_means.fit(X)
    # get predictions
    predictions = k_means.predict(X)

    # show the first 5 predictions
    X['Cluster'] = predictions
    print(X.head())

    # visualize 2 features of X after clustering
    sns.relplot(x="Longitude", y="Latitude", hue="Cluster", data=X, height=6)
    plt.show()


if __name__ == '__main__':
    # There are 2 examples:
    example_no = 0  # CHANGE THIS {0 for Image Compression | 1 for House Clustering}

    if example_no == 0:
        image_compression()
    elif example_no == 1:
        housing()
