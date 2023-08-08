import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import common as cm

N = 8
J = 1
h =	1
t_max = 100
NGrid = 200
r = np.sqrt(2**N)

dt = np.linspace(0., t_max, NGrid)
psi_0 = np.zeros((2**N))
psi_0[0] = 1

H = cm.h_tfim(h, J, N)
e, v = la.eigh(H)
v_dagg = np.transpose(np.conjugate(v))
S = np.zeros((NGrid),dtype=np.complex_)

for i in range(NGrid):
	U = np.dot(v, np.dot(np.diag(np.exp(-1j*e*dt[i])), v_dagg))
	psi_t = np.dot(U,psi_0)
	m = np.reshape(psi_t,(r,r))
	u,s,c = la.svd(m)
	for ii in s:
		S[i] = S[i]-(ii**2)*np.log(ii**2)

plt.plot(dt, S)
plt.xlabel("Time(t)")
plt.ylabel("Entropy s(t)")
plt.show()
