import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import common as cm

N = 8
J = 1
J_prime = 0.5
h =	1
d = 1
dh = 0.1
t_max = 100
NGrid = 300
assert N%2==0
r = 2**(N/2)

dt = np.linspace(0., t_max, NGrid)

psi_0 = np.zeros((2**N))
psi_0[0] = 1

#h_v = (2*np.random.rand(N)-1)
psi_0 = np.zeros((2**N))
psi_0[0] = 1
H = cm.h_tfim(h, J, N)
H_T = -cm.h_tfim(h+dh, J, N)
#H = cm.h_annni(h, J, J_prime, N)
#H_T = -cm.h_annni(h+dh, J, J_prime, N)
#psi_0 = cm.psi_0_xxz(N)
#H = cm.h_xxz(h, J, N, d)
#H_T = -cm.h_xxz(h+dh, J, N, d)

e, v = la.eigh(H)
v_dagg = np.transpose(np.conjugate(v))

psi_0_dagg = np.conjugate(np.transpose(psi_0))

m_z = cm.magnetization(N)
s_c = cm.spin_corr(N)
avg_m_z = np.zeros((NGrid))
S = np.zeros((NGrid))
avg_s_c = np.zeros((NGrid))
l = np.zeros((NGrid))

for i in range(NGrid):
	U = cm.exp_m(e,v,v_dagg,dt[i])
	psi_t = np.dot(U,psi_0)
	psi_t_dagg = np.transpose(np.conjugate(psi_t))
	avg_m_z[i] = cm.check_real(np.dot(psi_t_dagg,np.dot(m_z,psi_t)))
	avg_s_c[i] = cm.check_real(np.dot(psi_t_dagg,np.dot(s_c,psi_t)))
	S[i] = cm.entropy(psi_t,r)
	l[i] = cm.fidelity(psi_0_dagg,psi_t,N)

plt.subplot(4,1,1)
plt.plot(dt, avg_m_z)
plt.xlabel("Time(t)")
plt.ylabel("$<m_z>_t$")
plt.subplot(4,1,2)
plt.plot(dt, S)
plt.xlabel("Time(t)")
plt.ylabel("$S(t)$")
plt.subplot(4,1,3)
plt.plot(dt, avg_s_c)
plt.xlabel("Time(t)")
plt.ylabel("$s_c(t)$")
plt.subplot(4,1,4)
plt.plot(dt, l)
plt.xlabel("Time(t)")
plt.ylabel("$l(t)$")
plt.show()