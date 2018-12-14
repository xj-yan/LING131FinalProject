from abc import ABCMeta, abstractmethod

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
    def train(self): 
        pass


    @abstractmethod
    def classify(self, feature_set): 
        """
        :return: the most appropriate label for the given featureset.
        :rtype: label
        """
        pass

    # concrete method
    def mass_classify(self, feature_sets): 
        """
        :return: the most appropriate label for the given featureset.
        :rtype: list of labels
        """
        return [classify(self, fs) for fs in feature_sets]
    
    # concrete method
    def acurracy(self, test_set): 
        predicted = mass_classify(self, [test[0] for test in test_set])
        expected = [test[1] for test in test_set]
        count = 0
        for idx, val in enumerate(predicted):
            if expected[idx] == val:
                count += 1
        return float(count) / len(expected)
