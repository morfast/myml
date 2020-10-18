#!/usr/bin/env python3

import sys
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


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
    test()

