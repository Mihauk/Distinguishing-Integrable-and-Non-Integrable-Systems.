import numpy as np
import common as cm
from numpy import linalg as la

N = 2
h = 1
J = 1
t_max = 10
NGrid = 100
w1 = np.matrix([[1], [0]])
w2 = np.matrix([[0], [1]])
I = np.eye(2)

dt = np.linspace(0., t_max, NGrid)

psi_0 = np.zeros((2**N,1))
psi_0[0] = 1
psi_0_dagg = np.conjugate(psi_0.T)

H = cm.h_tfim(h, J, N)
e, v = la.eigh(H)
v_dagg = np.conjugate(v.T)

for i in range(NGrid):
	psi_t = cm.psi_at_t(e,v,v_dagg,dt[i],psi_0)
	psi_t_dagg = np.conjugate(psi_t.T)
	pho = np.dot(psi_t,psi_t_dagg)
	pho_a = np.dot(np.kron(np.conjugate(w1.T),I), np.dot(pho,np.kron(w1,I))) + np.dot(np.kron(np.conjugate(w2.T),I), np.dot(pho,np.kron(w2,I)))
	print(pho_a)