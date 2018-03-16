#!/usr/bin/python

""" 

My Logistic Regression 

step 0: initialize the weight vector and bias with zero.
step 1: compute the linear combination of the input features and weights.
step 2: apply the sigmoid activation function, which returns values between 0 and 1
step 3: compute the cost fucntion
step 4: compute the gradient of the cost function respect to the weights and bias
step 5: update the weights and bias

"""

import numpy as np

class LR:
    
    def __init__(self):
        # these parameters are determined in training and used in prediction
        self._weights_ = 0
        self._bias_ = 0
        self._trained_ = False

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def cross_entropy(y_truth, y_predict):
        """ cross entropy """
        s = np.sum(y_truth * np.log(y_predict) + (1.0 - y_truth) * (1.0 - np.log(y_predict)))
        return (-1.0 * s)/ y_truth.size

    def train(self, x, y_truth, max_iter, learning_rate):
        # initialize the parameters
        n_sample, n_feature = x.shape
        self._weights_ = np.zeros(n_feature)  # vector
        self._bias_ = 0.0  # scalar

        for n_iter in range(max_iter):
            # compute the linear combination, and apply the sigmoid function
            y_predict = self.sigmoid(np.dot(self._weights_, x) + self._bias_)

            # compute the loss
            loss = self.cross_entropy(y_truth, y_predict)
            print "iter: %d, loss: %f" % (n_iter, loss)

            if loss < 1.0e-4:
                break

            # compute the gradient
            dw = (1 / n_sample) * np.dot(x.T, (y_predict - y_true))
            db = (1 / n_sample) * np.sum(y_predict - y_true)

            # update the parameters
            self._weights_ -= learning_rate * dw
            self._bias_ -= learning_rate * db


    def predict(self, x, threshold = 0.50):
        if not self._trained_:
            print "Not trained"
            return

        y_predict = self.sigmoid(np.dot(self._weights_, x) + self._bias_)
        y_predict_labels = [ 1 if elem > threshold else 0 for elem in y_predict]

        return np.array(y_predict_labels)
