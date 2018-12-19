from abc import ABCMeta, abstractmethod
from models.AbstractClassifier import AbstractClassifier
import numpy as np

class NeuralNetwork(AbstractClassifier):
    """docstring for NeuralNetwork"""
    def __init__(self, x, y, dim2):
        """
        Initialization

        :param x: feature set
        :param y: label (class)
        :param dim2: hidden layer dimension

        fields:
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

    def labels(self):
        """
        :return a list of labels
        :return type list
        """
        return np.unique(self._labels)

    def sigmoid(self, x):
        """
        Calculate sigmoid
        :return: derivative
        :return type: array
        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        """
        Calculate derivative
        :param: x
        :param type: array
        :return: derivative
        :return type: array
        """
        return (1.0 - x) * x

    def backprop(self):
        """
        Back propogation
        :return: w1, w2
        :return type: float
        """
        layer_output_error = self._labels - self._output
        layer_output_delta = layer_output_error * self.derivative(self._output)
        layer_hidden_error = layer_output_delta.dot(self._weight2.T)
        layer_hidden_delta = layer_hidden_error * self.derivative(self._hidden_layer)
        w1 = self._input.T.dot(layer_hidden_delta)
        w2 = self._hidden_layer.T.dot(layer_output_delta)
        return w1, w2

    def feed_forward(self):
        """
        Feeding forward:
        update hidden layer and output layer
        :return: void
        """
        self._hidden_layer = self.sigmoid(np.dot(self._input, self._weight1))
        self._output = self.sigmoid(np.dot(self._hidden_layer, self._weight2))

    # override
    def train(self, number_of_iterations=None):
        """
        Train the data set
        :param: number_of_iterations
        :param type: int
        :return void
        :update _weight1 and _weight2
        """
        # interation count
        if number_of_iterations == None:
            number_of_iterations = 20000
        for i in range(number_of_iterations):
            # print("iteration", i)
            # feeding forward
            self.feed_forward()
            # back propagation
            w1, w2 = self.backprop()
            # update weight vector
            self._weight1 += w1
            self._weight2 += w2

    # override
    def classify(self, x_test):
        """
        Classify the input data
        :param: x_test 
        :param: input feature set for classification
        :return layer_output
        :return type: an array of output labels
        """
        layer_input = x_test
        layer_hidden = self.sigmoid(np.dot(layer_input,self._weight1))
        layer_output = self.sigmoid(np.dot(layer_hidden,self._weight2))
        return layer_output

    # override
    def accuracy(self, x_test, y_test):
        """
        Calculate the accuracy of test data
        :param: x_test 
        :param: input feature array for testing
        :param: y_test 
        :param: input label array for testing
        :return: the accuracy of the model
        :return type: a double in the range of [0, 100]
        """
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
