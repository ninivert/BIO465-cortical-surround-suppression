import numpy as np

__all__ = ['sigmoid', 'identity']

def sigmoid(x):
	return 0.5*(np.tanh(x) + 1)

def identity(x):
	return x