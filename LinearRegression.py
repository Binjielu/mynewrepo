import numpy as np
from h5py.h5s import NULL
import matplotlib.pyplot as plt
from numpy.core.umath import sign


def ols(y, y_hat):
    return 1 / (2 * len(y)) * np.sum((y - y_hat) ** 2)


def gaussian_radial_basis(x):
    phi = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        phi[:, i:i+1] = np.exp(-(0.5*(x-x[i]))**2)
    return phi


class LinearRegression:
    def __init__(self):
        self.w = NULL   # initialize weight

    def fit(self, x, y):
        # update weights with (x.t * x) * w = x.t * y => w = (x.t*x)^-1 * x.t*t
        self.w = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
        # print(self.w.shape)

    def predict(self, x):
        # predict y based on y = x * w
        y_hat = np.dot(x, self.w)
        return y_hat


class LinearRegressionGRB:
    def __init__(self):
        self.w = NULL     # initialize weight
        self.phi = NULL

    def fit(self, x, y):
        self.phi = gaussian_radial_basis(x)
        self.phi = np.hstack((np.ones((len(x), 1)), self.phi))
        self.w = np.linalg.solve((self.phi.T @ self.phi), (self.phi.T @ y))

    def predict(self, x):
        y_hat = self.phi @ self.w
        return y_hat


class LinearRegressionGradientDescent:
    def __init__(self):
        self.w = NULL

    def fit(self, x, y, lr=1e-3, epochs=1e3, show_curve=False):
        self.w = np.zeros((x.shape[1], y.shape[1]))
        losses = []
        for i in range(int(epochs)):
            y_hat = self.predict(x)
            loss = ols(y, y_hat)
            losses.append(loss)
            self.w -= lr * 1 / len(x) * x.T @ (y_hat - y)
        if show_curve:
            plt.plot(epochs, losses)
            plt.xlabel("Epochs")
            plt.ylabel("loss")
            plt.title("Training Curve")

    def predict(self, x):
        y_hat = x @ self.w
        return y_hat


class LinearRegressionGradientDescentRegularization:
    def __init__(self):
        self.w = NULL
        self.b = NULL

    def fit(self, x, y, lr=1e-3, epochs=1e3, reg_rate=5, ratio=0.1, show_curve=False):
        n = len(x)
        lambda1 = reg_rate * ratio
        lambda2 = reg_rate * (1 - ratio)

        self.w = np.zeros((x.shape[1], y.shape[1]))
        self.b = np.zeros((1, y.shape[1]))
        losses = []
        for i in range(int(epochs)):
            y_hat = self.predict(x)
            loss = ols(y, y_hat) + lambda1 / (2 * n) * np.sum(abs(self.w)) + lambda2 / 2 * np.sum(self.w**2)
            losses.append(loss)
            self.w -= lr * (1 / n * x.T @ (y_hat - y) + lambda1 * sign(self.w) + lambda2 * self.w)
            self.b -= lr * (1 / n * np.sum(y_hat - y))
        if show_curve:
            plt.plot(epochs, losses)
            plt.xlabel("Epochs")
            plt.ylabel("loss")
            plt.title("Training Curve")

    def predict(self, x):
        y_hat = x @ self.w + self.b
        return y_hat
