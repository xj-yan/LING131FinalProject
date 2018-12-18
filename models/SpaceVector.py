from abc import ABCMeta, abstractmethod
from AbstractClassifier import AbstractClassifier 

class SpaceVector(object):
	"""docstring for SpaceVector"""
	def __init__(self, arg):
		super(SpaceVector, self).__init__()
		self.arg = arg
		