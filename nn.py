#!/usr/bin/python

import numpy as np
import pandas as pd

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


def main():
    x, y = make_circles(n_samples = 1000, factor = 0.5, noise = 0.1)
    y_true = y[:, np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(x, y_true)

class NN():
    
    def __init__(self, n_input, n_output, n_hidden):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden

        # weights of the hidden layer
        self.W_h = np.random.randn(self.n_input, self.hidden)
        # bias of the hidden layer
        self.b_h = np.zeros((1, self.hidden))

        # weights of the output layer
        self.W_h = np.random.randn(self.hidden, self.output)
        # bias of the output layer
        self.b_h = np.zeros((1, self.output))

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))



