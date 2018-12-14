import abc
from abc_base import AbstractClassifier

class AdaBoost(AbstractClassifier):
	"""docstring for AdaBoost"""
	def __init__(self, arg):
		super(AdaBoost, self).__init__()
		self.arg = arg
		