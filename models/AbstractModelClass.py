from abc import ABCMeta, abstractmethod

class AbstractModelClass(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, training_set): pass

    # static method
    def acurracy(self, test_set): pass

    # static method
    def classify(self, featureSet): pass