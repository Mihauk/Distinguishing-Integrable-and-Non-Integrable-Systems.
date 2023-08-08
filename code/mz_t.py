import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import common as cm

N = 10
J = 1
h =	1
t_max = 100
NGrid = 1000

dt = np.linspace(0., t_max, NGrid)
psi_0 = np.zeros((2**N))
psi_0[0] = 1

h_v = (2*np.random.rand(N)-1)

H = cm.h_tfim(h, J, N)
H1 = cm.h_tfim(h, J, N, h_v)
e, v = la.eigh(H)
e1, v1 = la.eigh(H1)
v_dagg = np.transpose(np.conjugate(v))
v_dagg1 = np.transpose(np.conjugate(v1))
m_z = cm.magnetization(N)
avg_m_z = np.zeros((NGrid))
avg_m_z1 = np.zeros((NGrid))

for i in range(NGrid):
	psi_t = cm.psi_at_t(e,v,v_dagg,dt[i],psi_0)
	psi_t1 = cm.psi_at_t(e1,v1,v_dagg1,dt[i],psi_0)
	psi_t_dag = np.transpose(np.conjugate(psi_t))
	psi_t_dag1 = np.transpose(np.conjugate(psi_t1))
	avg_m_z[i] = cm.check_real(np.dot(psi_t_dag,np.dot(m_z,psi_t)))
	avg_m_z1[i] = cm.check_real(np.dot(psi_t_dag1,np.dot(m_z,psi_t1)))

plt.plot(dt, avg_m_z, label=r"$h_v=0$")
plt.plot(dt, avg_m_z1, label=r"$h_v=[-1,1]$")
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle m_z \rangle$", fontsize=18)
plt.legend()
plt.show()