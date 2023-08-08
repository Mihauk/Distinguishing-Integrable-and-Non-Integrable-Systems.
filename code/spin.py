import numpy as np

def spin(A, i, N):
	assert (i<N), "specify the boundary condition"
	assert (A=="X" or A=="Y" or A=="Z"), "Specify the corect pauli spin matrices from X, Y and Z"
	s=1
	if A=='X':
		spin=s_x
	elif A=='Y':
		spin=s_y
	elif A=='Z':
		spin=s_z
	for ii in range(N):
		if i==ii:
			s=np.kron(s,spin)
		else :
			s=np.kron(s,s_0)
	return s

s_0 = np.array([[1, 0],[0, 1]])
s_x = .5*np.array([[0, 1],[1, 0]])
s_y = .5*np.array([[0, -1j],[1j, 0]])
s_z = .5*np.array([[1, 0],[0, -1]])