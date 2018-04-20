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


    def forward_pass(self, x):
        """ input: x, propagates the input x forward through the net """
        # hidden units
        A_h = np.dot(x, self.W_h) + self.b_h
        O_h = np.tanh(A_h)

        # output units
        A_o = np.dot(O_h, self.W_o) + self.b_o
        O_o = self.sigmoid(A_o)

        outputs = {
            "A_h": A_h,
            "A_o": A_o,
            "O_h": O_h,
            "O_o": O_o,
        }

        return outputs

    def cost(self, y_true, y_predict, n_samples):
        cost = (-1 / n_samples) * np.sum(y_true * np.log(y_predict) + \
               (1-y_true) * (np.log(1 - y_predict)))
        cost = np.squeeze(cost)

        return cost

    def backward_pass(self, x, y, n_samples, outputs):
        pass

    def update_weights(self, gradient, learning_rate):
        pass


    def train(self, x, y, n_iter, learning_rate):
        pass

    def predict(self, x):
        pass









