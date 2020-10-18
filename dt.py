#!/usr/bin/env python3

import sys
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from scipy.stats import entropy as entropy2


def entropy(freq):
    f = np.array(freq)
    s = np.sum(f)
    f = np.divide(f, s)

    result = 0.0
    for felem in f:
        if np.isclose(felem, 0.0):
            continue
        result += (-felem * np.log2(felem))
    return result


def test_entropy():
    array = np.random.randint(10, size = 100)
    r1 = entropy(array)
    r2 = entropy2(array, base=2.0)
    if (np.isclose(r1, r2)):
        print("test_entropy pass")


class myDesicionTree(object):
    def __init__(self):
        pass


    def fit(self, x_train, y_train):
        pass


    def predict(self, x_test):
        pass


def test():
    x, y_true = make_blobs(n_samples=300, centers=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y_true)

    model = myDesicionTree()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    error = np.mean(np.abs(y_test - y_predict))
    print(error)


if __name__ == '__main__':
    test_entropy()


