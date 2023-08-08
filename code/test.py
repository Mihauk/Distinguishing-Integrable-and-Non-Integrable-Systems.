import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
#from sympy.physics.matrices import msigma

def spinz(i):
	s=1
	for ii in range(N):
		if i==ii:
			s=np.kron(s,s_z)
		else :
			s=np.kron(s,s_0)
	return s

def spinx(i):
	s=1
	for ii in range(N):
		if i==ii:
			s=np.kron(s,s_x)
		else :
			s=np.kron(s,s_0)
	return s

def spiny(i):
	s=1
	for ii in range(N):
		if i==ii:
			s=np.kron(s,s_y)
		else :
			s=np.kron(s,s_0)
	return s

def hermitian(x):
	y=x
	y=np.conjugate(y)
	y=np.transpose(y)
	if np.allclose(x,y):
		print("Hermitian")
	else:
		print("Not Hermitian")

def hamiltonian(xVal):
	x=0
	for i in range(N):
		x = x - J*spinz(i)*spinz((i+1)%N) + xVal*spinx(i) - 1e-3*spinz(i)
	return(x)

def magnetization():
	x=0
	for i in range(N):
		x = x + (1./N)*spinz(i)
	return(x)

s_0 = np.array([[1, 0],[0, 1]])
s_x = .5*np.array([[0, 1],[1, 0]])
s_y = .5*np.array([[0, -1j],[1j, 0]])
s_z = .5*np.array([[1, 0],[0, -1]])

'''p_0 = np.eye(2)
p_x = msigma(1)
p_y = msigma(2)
p_z = msigma(3)'''

N = 6
J = 1
h = 1
NGird = 100
xVal=np.linspace(0., h, NGird)

m_z = magnetization()
avg_m_z = np.zeros((NGird))

for i in range(NGird):
	H = hamiltonian(xVal[i])
	#hermitian(H)
	e, v = la.eigh(H)
	#print e[:5]
	psi_0 = v[:,0]
	psi_0_t = np.transpose(psi_0)
	psi_0_conj=np.conjugate(psi_0_t)
	#print(np.dot(psi_0, psi_0_conj))
	avg_m_z[i] = np.dot(psi_0_conj,np.dot(m_z,psi_0))

plt.plot(xVal, avg_m_z)
plt.xlabel("Magnetic Field(h)")
plt.ylabel("Magnetization <m_z>_o")
plt.show()