import numpy as np
from numpy import linalg as la
import math

def exponential(x, N):
	e = 0
	s = np.eye(2**N)
	i = 1
	while (i < 100):
		e=e+s
		s=la.matrix_power(x, i)/math.factorial(i)
		i=i+1
	return e