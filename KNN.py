import numpy as np


# weight function
def uniform_weights(dists, *args):
    K = len(dists)
    return np.array([1 / K] * K)


def reciprocal_weights(dists, epsilon=1e-2):
    return 1 / (np.sqrt(dists) + epsilon)


def gauss_weights(dists, *args):
    ed = np.exp(dists)
    return ed / np.sum(ed)


class KNN:
    def __init__(self, K, p=2, weight_function=reciprocal_weights, epsilon=1e-2, mode=1):
        self.mode = mode
        self.K = K    # number of nearest neighbors
        self.p = p    # norm selection
        self.weight_function = weight_function    # weight function based on distance
        self.epsilon = epsilon     # to prevent get the infinite weight from weight function

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        n = len(x)     # number of observations

        y_hat = np.zeros(n)

        for i in range(n):
            # calculate distance between selected observation and every other observations
            dists = np.sum((self.x - x[i]) ** 2, axis=1)
            # sort the distance and select the k nearest neighbors's index
            idx = dists.argsort()[:self.K]
            # sign weight on each neighbor based on their distance
            gamma = self.weight_function(dists[idx], self.epsilon)

            if self.mode:
                y_hat[i] = gamma.dot(self.y[idx]) / gamma.sum()
            else:
                # get the index of the maximum bin counts
                y_hat[i] = np.bincount(self.y[idx], weights=gamma).argmax()
            # y_hat[i] = np.bincount(self.y[idx]).argmax()

        return y_hat



