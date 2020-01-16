import numpy as np
from matplotlib import pyplot as plt


# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class Tanh:
    def __call__(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def D(self, z):
        return 1 - (self(z) * self(z))


class ReLu:
    def __call__(self, z):
        return z * (z > 0)

    def D(self, z):
        return z > 0


class Sigmoid:
    def __inti__(self):
        pass

    def __call__(self, z):  # no need to specify the function name, class will automatically call this function
        return 1 / (1 + np.exp(-z))

    def D(self, z):
        return self(z) * (1 - self(z))


class BinaryLogisticRegression:
    def __init__(self, shape_in):
        self.weights = np.random.randn(shape_in, 1)
        self.bias = np.random.randn(1, 1)

    def fit(self, x, y, learn_rate=1e-3, iteration=1000):
        losses = []
        for i in range(iteration):
            y_hat = x @ self.weights + self.bias
            p_hat = sigmoid(y_hat)

            loss = -np.sum(y * np.log(p_hat + 1e-50) + (1 - y) * np.log(1 - p_hat + 1e-50))
            losses.append(loss)

            self.weights -= learn_rate * x.T @ (p_hat - y)
            self.bias -= learn_rate * np.sum(p_hat - y, axis=0)
        plt.plot(losses)
        plt.show()

    def predict(self, x):
        y_hat = x @ self.weights + self.bias
        p_hat = sigmoid(y_hat)
        return p_hat


class MultipleLogisticRegression:
    def __init__(self, shape_in, shape_out):
        self.weights = np.random.randn(shape_in, shape_out)
        self.bias = np.random.randn(1, shape_out)

    def fit(self, x, y, learn_rate=1e-3, iteration=1000):
        losses = []
        for i in range(iteration):
            y_hat = x @ self.weights + self.bias
            p_hat = softmax(y_hat)

            loss = -np.sum(y * np.log(p_hat + 1e-50))
            losses.append(loss)

            self.weights -= learn_rate * x.T @ (p_hat - y)
            self.bias -= learn_rate * np.sum(p_hat - y, axis=0)
        plt.plot(losses)

    def batch_fit(self, batch_x, batch_y, learn_rate=1e-3):
        y_hat = batch_x @ self.weights + self.bias
        p_hat = softmax(y_hat)

        loss = -np.sum(batch_y * np.log(p_hat + 1e-50))

        self.weights -= learn_rate * batch_x.T @ (p_hat - batch_y)
        self.bias -= learn_rate * np.sum(p_hat - batch_y, axis=0)
        return loss

    def predict(self, x):
        y_hat = x @ self.weights + self.bias
        p_hat = softmax(y_hat)
        return p_hat
