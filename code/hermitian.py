import numpy as np

def hermitian(x):
	y=x
	y=np.conjugate(np.transpose(y))
	if np.allclose(x,y):
		print("Hermitian")
	else:
		print("Not Hermitian")