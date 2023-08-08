import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import common as cm

N = 6
#N1 = 8
#N2 = 10
J = 1
#Jp=0.5
h = 1
NGrid = 100
dh = np.linspace(0., h, NGrid)

m_z = cm.magnetization(N)
#m_z1 = cm.magnetization(N1)
#m_z2 = cm.magnetization(N2)
avg_m_z = np.zeros((NGrid))
#avg_m_z1 = np.zeros((NGrid))
#avg_m_z2 = np.zeros((NGrid))

for i in range(NGrid):
	H = cm.h_tfim(dh[i],J,N)
	H1 = cm.h_tfim(dh[i],J,N1)
	H2 = cm.h_tfim(dh[i],J,N2)
	#H = cm.h_annni(dh[i],J,Jp,N)
	#H1 = cm.h_annni(dh[i],J,Jp,N1)
	#H2 = cm.h_annni(dh[i],J,Jp,N2)
	e, v = la.eigh(H)
	#e1, v1 = la.eigh(H1)
	#e2, v2 = la.eigh(H2)
	psi_0 = v[:,0]
	#psi_01 = v1[:,0]
	#psi_02 = v2[:,0]
	psi_0_dag = np.transpose(np.conjugate(psi_0))
	#psi_0_dag1 = np.transpose(np.conjugate(psi_01))
	#psi_0_dag2 = np.transpose(np.conjugate(psi_02))
	avg_m_z[i] = cm.check_real(np.dot(psi_0_dag,np.dot(m_z,psi_0)))
	#avg_m_z1[i] = cm.check_real(np.dot(psi_0_dag1,np.dot(m_z1,psi_01)))
	#avg_m_z2[i] = cm.check_real(np.dot(psi_0_dag2,np.dot(m_z2,psi_02)))

plt.plot(dh, avg_m_z, label=r"$N=6$")
#plt.plot(dh, avg_m_z1, label=r"$N=8$")
#plt.plot(dh, avg_m_z2, label=r"$N=10$")
plt.xlabel(r"$h$", fontsize=18)
plt.ylabel(r"$\langle m_z \rangle_o$", fontsize=18)
plt.legend()
plt.show()