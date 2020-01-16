import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def Encode(name):
    name = ''.join(['{', name.lower(), '|'])
    out = []
    for char in name:
        char_encoded = torch.zeros((1, 28))
        char_encoded[0, ord(char) - 'a'] = 1
        out.append(char_encoded)
    return torch.cat(out, dim=0)


def Decode(tensor):
    ords = torch.argmax(tensor, dim=1)
    out = []
    for i in ords:
        out.append(chr(i + 'a'))
    return ''.join(out)


class Layer(torch.nn.Module):
    def __init__(self, size_in, size_out, activation_func=torch.sigmoid):
        super(Layer, self).__init__()
        self.w = torch.nn.Parameter(
            torch.randn(size_in, size_out, requires_grad=True))
        self.b = torch.nn.Parameter(
            torch.randn(1, size_out, requires_grad=True))
        self.activation_func = activation_func

    def Forward(self, x):
        return self.activation_func(x @ self.w + self.b)


class RNN(torch.nn.Module):
    def __init__(self, size_in, size_out, size_mem):
        super(RNN, self).__init__()
        self.layer_0 = Layer(size_in + size_mem, size_mem)
        self.layer_out = Layer(size_mem, size_out, lambda x: x)
        self.size_mem = size_mem

    def Forward(self, x):
        mem = torch.zeros((1, self.size_mem))
        out = []
        for i in range(x.shape[0]):
            z = torch.cat([x[[i], :], mem], dim=1)
            mem = self.layer_0.Forward(z)
            out.append(self.layer_out.Forward(mem))
        out = torch.cat(out, dim=0)
        return out

    def Generate(self, start, iterations):
        mem = torch.randn((1, self.size_mem))
        out = [start]
        for i in range(iterations):
            z = torch.cat([out[i], mem], dim=1)
            mem = self.layer_0.Forward(z)
            out.append(self.layer_out.Forward(mem))
            if Decode(out[-1]) == '|':
                break

        out = torch.cat(out, dim=0)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, size_in, size_out, size_short, size_long):
        super(LSTM, self).__init__()
        self.layer_0 = Layer(size_in + size_short, size_long)
        self.layer_1 = Layer(size_in + size_short, size_long)
        self.layer_2 = Layer(size_in + size_short, size_long, activation_func=torch.tanh)
        self.layer_3 = Layer(size_in + size_short, size_short)
        self.layer_4 = Layer(size_long, size_short, activation_func=torch.tanh)
        self.layer_out = Layer(size_short, size_out, lambda x: x)
        self.size_short = size_short
        self.size_long = size_long

    def Forward(self, x):
        short_mem = torch.zeros(1, self.size_short)
        long_mem = torch.zeros(1, self.size_long)
        out = []
        for i in range(x.shape[0]):
            z = torch.cat([x[[i], :], short_mem], dim=1)
            long_mem = self.layer_0.Forward(z) * long_mem + self.layer_1.Forward(z) * self.layer_2.Forward(z)
            short_mem = self.layer_3.Forward(z) * self.layer_4.Forward(long_mem)
            out.append(self.layer_out.Forward(short_mem))
        out = torch.cat(out, dim=0)
        return out

    def Generate(self, start, iteration):
        short_mem = torch.randn(1, self.size_short)
        long_mem = torch.randn(1, self.size_long)
        out = [start]
        for i in range(iteration):
            z = torch.cat([out[i], short_mem], dim=1)
            long_mem = self.layer_0.Forward(z) * long_mem + self.layer_1.Forward(z) * self.layer_2.Forward(z)
            short_mem = self.layer_3.Forward(z) * self.layer_4.Forward(long_mem)
            out.append(self.layer_out.Forward(short_mem))
            if Decode(out[-1]) == '|' or len(Decode(out[-1])) > 30:
                break

        out = torch.cat(out, dim=0)
        return out



