from abc import ABCMeta, abstractmethod
from models.AbstractClassifier import AbstractClassifier
import numpy as np

class NaiveBayes(AbstractClassifier):
    """
    docstring for NaiveBayes
    
    """
    # override
    def __init__(self, x, y, alpha = 1.0):
        """
        Intitialization
        :param x: input feature array
        :param y: input class array
        :alpha: smoothing
        _alpha: smoothing

        """
        self._alpha = alpha
        self._x = x
        self._y = y
        self._labels = np.unique(y)

    def labels(self):
        """
        :return a list of labels
        :return type list
        """
        return self._labels

    # override
    def train(self):
        """
        Train the data set
        :return self
        """
        x = self._x
        y = self._y
        # group the array by class
        group_by_class = []
        for c in self._labels:
            group_by_class.append([a for a, b in zip(x, y) if b == c])
        total_length = x.shape[0]
        # probability by class
        self._prob_by_class = [np.log(len(i)) - np.log(total_length) for i in group_by_class]
        count_by_class = np.array([np.array(i).sum(axis = 0) for i in group_by_class]) + self._alpha
        # probability of each feature given class
        self._prob_by_feature = np.log(count_by_class) - np.log(count_by_class.sum(axis = 1)[np.newaxis].T)
        return self

    # override
    def classify(self, x_test):
        """
        Classify the data set
        :param x_set: an array of feature set
        :return: most possible labels
        """
        prob_by_testset = [self._prob_by_class + (self._prob_by_feature * i).sum(axis = 1) for i in x_test]
        return np.argmax(prob_by_testset)

    def accuracy(self, x, y):
        """
        Calculate the accuracy of trained model
        :param x : an array of feature set
        :param y: an array of corresponding labels
        :return: percentage of correct prediction
        """
        length = y.shape[0]
        x = self.classify(x)
        return (100.0 * (x == y).sum(axis = 0) / length)[0]
        

    


