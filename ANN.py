import numpy as np
import matplotlib.pyplot as plt


# Activation function

class Sigmoid:
    def __inti__(self):
        pass

    def __call__(self, z):  # no need to specify the function name, class will automatically call this function
        return 1 / (1 + np.exp(-z))

    def D(self, z):
        return self(z) * (1 - self(z))


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


# ======

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def Xline(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_line = np.random.rand(10000, 2) * (x_max - x_min) + x_min
    return x_line


class ANN:
    # two layer log reg
    def __init__(self, size_in, size_out, size_hidden, activation_func):
        self.func = activation_func

        self.w_1 = np.random.randn(size_hidden, size_out)
        self.b_1 = np.random.randn(1, size_out)

        self.w_0 = np.random.randn(size_in, size_hidden)
        self.b_0 = np.random.randn(1, size_hidden)

    def fit(self, x, y, learn_rate=1e-3, iteration=1000):
        losses = []
        for i in range(iteration):
            z = x @ self.w_0 + self.b_0
            sig = self.func(z)
            p_hat = softmax(sig @ self.w_1 + self.b_1)

            grad_p_hat = p_hat - y

            grad_w_1 = sig.T @ grad_p_hat
            grad_b_1 = np.sum(grad_p_hat, axis=0, keepdims=True)

            grad_w_0 = x.T @ (grad_p_hat @ self.w_1.T * self.func.D(z))
            grad_b_0 = np.sum(grad_p_hat @ self.w_1.T * self.func.D(z), axis=0, keepdims=True)

            self.w_1 -= grad_w_1 * learn_rate
            self.b_1 -= grad_b_1 * learn_rate
            self.w_0 -= grad_w_0 * learn_rate
            self.b_0 -= grad_b_0 * learn_rate
            losses.append(-np.sum(y * np.log(p_hat + 1e-50)))

        plt.plot(losses)

    def predict(self, x):
        z = self.func(x @ self.w_0 + self.b_0)
        p_hat = softmax(z @ self.w_1 + self.b_1)
        return p_hat

    def batch_fit(self, x, y):
        pass


class ThreeLayerANN:
    # three layer log reg
    def __init__(self, size_in, size_out, z_0_size, z_1_size, activation_func):
        self.func_0 = activation_func
        self.func_1 = activation_func

        self.w_2 = np.random.randn(z_1_size, size_out)
        self.b_2 = np.random.randn(1, size_out)

        self.w_1 = np.random.randn(z_0_size, z_1_size)
        self.b_1 = np.random.randn(1, z_1_size)

        self.w_0 = np.random.randn(size_in, z_0_size)
        self.b_0 = np.random.randn(1, z_0_size)

    def fit(self, x, y, learn_rate=1e-3, iteration=1000):
        losses = []
        for i in range(iteration):
            z_0 = x @ self.w_0 + self.b_0
            sig_0 = self.func_0(z_0)
            z_1 = sig_0 @ self.w_1 + self.b_1
            sig_1 = self.func_1(z_1)
            p_hat = softmax(sig_1 @ self.w_2 + self.b_2)

            grad_p_hat = p_hat - y

            grad_w_2 = sig_1.T @ grad_p_hat
            grad_b_2 = np.sum(grad_p_hat, axis=0, keepdims=True)

            grad_w_1 = sig_0.T @ (grad_p_hat @ self.w_2.T * self.func_1.D(z_1))
            grad_b_1 = np.sum(grad_p_hat @ self.w_2.T * self.func_1.D(z_1), axis=0, keepdims=True)

            grad_w_0 = x.T @ ((((grad_p_hat @ self.w_2.T) * self.func_1.D(z_1)) @ self.w_1.T) * self.func_0.D(z_0))
            grad_b_0 = np.sum((((grad_p_hat @ self.w_2.T) * self.func_1.D(z_1)) @ self.w_1.T) * self.func_0.D(z_0),
                              axis=0, keepdims=True)

            self.w_2 -= grad_w_2 * learn_rate
            self.b_2 -= grad_b_2 * learn_rate
            self.w_1 -= grad_w_1 * learn_rate
            self.b_1 -= grad_b_1 * learn_rate
            self.w_0 -= grad_w_0 * learn_rate
            self.b_0 -= grad_b_0 * learn_rate
            losses.append(-np.sum(y * np.log(p_hat + 1e-99)))

        plt.plot(losses)

    def predict(self, x):
        z_0 = x @ self.w_0 + self.b_0
        sig_0 = self.func_0(z_0)
        z_1 = sig_0 @ self.w_1 + self.b_1
        sig_1 = self.func_1(z_1)
        p_hat = softmax(sig_1 @ self.w_2 + self.b_2)
        return p_hat

    def batch_fit(self, x, y):
        pass
