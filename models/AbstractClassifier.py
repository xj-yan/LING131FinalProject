from abc import ABCMeta, abstractmethod

class AbstractClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def labels(self): 
        """
        :return: list of labels
        :rtype: list of (immutable)
        """
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def classify(self, feature_set): 
        """
        :return: the most appropriate label for the given featureset.
        :rtype: label
        """
        pass
        
    @abstractmethod
    def accuracy(self, x, y):
        pass
        
