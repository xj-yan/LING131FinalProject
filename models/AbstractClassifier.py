from abc import ABCMeta, abstractmethod
from collections import defaultdict

class AbstractClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """
        abstract class should not be able to be initialized
        """
        pass

    @abstractmethod
    def labels(self): 
        """
        :return: the list of labels
        :rtype: list of (immutable)
        """
        pass

    @abstractmethod
    def train(self, labeled_feature_set): 
        pass

    @abstractmethod
    def classify(self, feature_set): 
        """
        :return: the most appropriate label for the given featureset.
        :rtype: label
        """
        pass

    @abstractmethod
    def mass_classify(self, feature_sets): 
        """
        :return: the most appropriate label for the given featureset.
        :rtype: list of labels
        """
        return [self.classify(fs) for fs in feature_sets]
    
    @abstractmethod
    def accuracy(self, test_set): 
        count = 0
        total = 0
        # print(test_set[1][1], self.classify(test_set[1][0]))
        # return 0
        for test in test_set:
            fff = self.classify(test[0])
            print(test[1], fff)
            if test[1] == fff:
                count += 1
            total += 1
        return float(count) / total
        
