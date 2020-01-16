import torch
from torchvision.datasets import MNIST
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def Xline(x, d):
    x_max = torch.max(x, dim=0)[0]
    x_min = torch.min(x, dim=0)[0]
    x_line = torch.rand(10000, d) * (x_max - x_min) + x_min
    return x_line


class Layer(torch.nn.Module):
    def __init__(self, size_in, size_out, activation_func=torch.relu, dropout_rate=0, noise_rate=0):
        super(Layer, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(size_in, size_out, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn(1, size_out, requires_grad=True))
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.noise_rate = noise_rate

    def forwards(self, x):
        z = self.activation_func(x @ self.w + self.b)
        noise = torch.randn(z.shape) * self.noise_rate
        mask = torch.rand(z.shape) > self.dropout_rate
        return (z + noise) * mask

    def predicts(self, x):
        # do predict with the data without the dropout
        return self.activation_func(x @ self.w + self.b)


class ThreeLayerANN(torch.nn.Module):
    def __init__(self, size_in, size_hidden_0, size_hidden_1, size_out):
        super(ThreeLayerANN, self).__init__()
        self.layer0 = Layer(size_in, size_hidden_0, noise_rate=0, dropout_rate=0)
        self.layer1 = Layer(size_hidden_0, size_hidden_1)
        self.layer2 = Layer(size_hidden_1, size_out)

    def forwards(self, x):
        return self.layer2.forwards(
            self.layer1.forwards(
                self.layer0.forwards(x)))

    def predicts(self, x):
        return self.layer2.predicts(
            self.layer1.predicts(
                self.layer0.predicts(x)))


class MultiLayerANN(torch.nn.Module):
    def __init__(self, sizes):
        super(MultiLayerANN, self).__init__()
        self.layers = []
        for i in range(1, len(sizes)):
            new_layer = Layer(sizes[i - 1], sizes[-1])
            # save layer as atrributes
            setattr(self, 'layer_{}'.format(i), new_layer)
            self.layers.append(new_layer)

    def forwards(self, x):
        z = x
        for layer in self.layer:
            z = layer.forwards(x)
        return z

    def predicts(self, x):
        return self.layer2.predicts(
            self.layer1.predicts(
                self.layer0.predicts(x)))


class MultiLayerBypassANN(torch.nn.Module):
    def __init__(self, size_in, size_out, sizes, num):
        super(MultiLayerBypassANN, self).__init__()

        #         sizes = sizes[-1] * 2 + sizes
        self.layer_in = Layer(size_in, sizes[0])
        self.models = []
        self.layers = []

        for i in range(num):
            new_model = MultiLayerANN(sizes)
            # save layer as atrributes
            setattr(self, 'model_{}'.format(i), new_model)
            self.models.append(new_model)
            new_layer = Layer(sizes[-1] + sizes[0], sizes[0])
            setattr(self, 'layer_{}'.format(i), new_layer)
            self.layers.append(new_layer)

        self.layer_out = Layer(sizes[0], size_out)
        self.num = num

    def forwards(self, x):
        z = self.layer_in.forwards(x)

        for i in range(self.num):
            z_new = self.models[i].fowards(z)
            z = torch.cat([z, z_new], dim=1)
            z = self.layers[i].forwards(z)
        z = self.layer_out.forwards(z)

        return z
