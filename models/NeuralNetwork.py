from abc import ABCMeta, abstractmethod
from AbstractClassifier import AbstractClassifier
import numpy as np

class NeuralNetwork(AbstractClassifier):
    """docstring for NeuralNetwork"""
    def __init__(self, x, y, dim2):
        """
        Intitialization
        _input: feature sets
           type: array
        _labels: labels
           type: array
        _output: 
           type: array
        _weight1: input layer -> hidden layer
           type: array
        _weight2: hidden layer -> output layer
           type: array
        """
        self._input = x
        self._labels = y
        self._output = np.zeros(y.shape)
        self._weight1 = np.random.random((x.shape[1], dim2))
        self._weight2 = np.random.random((dim2, 1))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        return (1.0 - x) * x

    def backprop(self):
        layer_output_error = self._labels - self._output
        layer_output_delta = layer_output_error * self.derivative(self._output)
        layer_hidden_error = layer_output_delta.dot(self._weight2.T)
        layer_hidden_delta = layer_hidden_error * self.derivative(self._hidden_layer)
        w1 = self._input.T.dot(layer_hidden_delta)
        w2 = self._hidden_layer.T.dot(layer_output_delta)
        return w1, w2

    def feed_forward(self):
        self._hidden_layer = self.sigmoid(np.dot(self._input, self._weight1))
        self._output = self.sigmoid(np.dot(self._hidden_layer, self._weight2))

    def train(self, number_of_iterations=None):
        # interations
        if number_of_iterations == None:
            number_of_iterations = 20000
        for i in range(number_of_iterations):
            # feeding forward
            self.feed_forward()
            # back propagation
            w1, w2 = self.backprop()
            # update weight vector
            self._weight1 += w1
            self._weight2 += w2

    def classify(self, x_test):
        layer_input = x_test
        layer_hidden = self.sigmoid(np.dot(layer_input,self._weight1))
        layer_output = self.sigmoid(np.dot(layer_hidden,self._weight2))
        return layer_output

    def accuracy(self, x_test, y_test):
        layer_output = self.classify(x_test)
        count = 0
        for i in range(len(layer_output)):
            if layer_output[i][0] < 0.5:
                layer_output[i][0] = 0
            else:
                layer_output[i][0] = 1
            if(layer_output[i][0] == y_test[i][0]):
                count += 1
        return count * 100.0 / len(layer_output)
