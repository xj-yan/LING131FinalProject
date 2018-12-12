import abc
from abc_base import AbstractModelClass

class NeuralNetwork(AbstractModelClass):
	"""docstring for NeuralNetwork"""
	def __init__(self, arg):
		super(NeuralNetwork, self).__init__()
		self.arg = arg
		
	def train(self, training_set):
		
		