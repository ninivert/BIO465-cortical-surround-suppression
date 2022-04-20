import scipy.integrate
import numpy as np
from dataclasses import dataclass
from typing import Callable

__all__ = ['Populations']

@dataclass
class Populations:
	# h: np.ndarray  # input potentials at time t. shape=(K,)
	W: np.ndarray  # interaction weights. shape=(K, K)
	tau: np.ndarray  # time constant of population activity
	R: float  # resistivity
	gain_fn: Callable[[np.ndarray], np.ndarray]  # F(h) : R^K -> R^K, gain function of each population
	I_ext: Callable[[float], np.ndarray]  # I_ext(t) : R+ -> R^K, external stimulus
	# ASSUMPTION : filter_fn is a dirac delta
	# filter_fn: Callable[[float], np.ndarray]  # alpha(s) : R -> R^K, filter function of each population

	def dh(self, t: float, h: np.ndarray):
		rhs = np.zeros_like(h)
		
		# Exponential decay
		rhs -= h
		
		# Network currents
		gain = self.gain_fn(h)
		filtered = gain  # filter_fn is a dirac delta, normally this is in integral form
		I_network = np.einsum('kn,n->k', self.W, filtered)
		rhs += self.R * I_network

		# External currents
		rhs += self.R * self.I_ext(t)

		# Apply tau
		rhs /= self.tau

		return rhs

	def simulate_h(self, h0: np.ndarray, t_span: tuple[float, float], dt_max: float = 0.1):
		res = scipy.integrate.solve_ivp(self.dh, t_span, h0, max_step=dt_max)
		return res