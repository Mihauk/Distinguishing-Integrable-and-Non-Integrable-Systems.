import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import common as cm

def time_evol(e,v,v_dagg,t,psi_0,psi_0_dagg):
	psi_t = cm.psi_at_t(e,v,v_dagg,t,psi_0)
	psi_t_dagg = np.transpose(np.conjugate(psi_t))
	a = cm.check_real(np.dot(psi_t_dagg,np.dot(m_z,psi_t)))
	b = cm.check_real(np.dot(psi_t_dagg,np.dot(s_c,psi_t)))
	c = cm.bipartite_fl(s_l2, psi_t, psi_t_dagg)
	d = cm.entropy(psi_t,r)
	e = cm.fidelity(psi_0_dagg,psi_t,N)
	return(a,b,c,d,e)

N = 10
J = 1
J_prime = 1
h =	1
#d = 1
dh = 0.1
tau=100
t_max = 2.5*tau
NGrid = 300
assert N%2==0
r = int(2**(N/2))

dt = np.linspace(0., t_max, NGrid)

m_z = cm.magnetization(N)
s_c = cm.spin_corr(N)
s_l2 = cm.sl2(N)

avg_m_z = np.zeros((NGrid))
S = np.zeros((NGrid))
b_fl = np.zeros((NGrid))
avg_s_c = np.zeros((NGrid))
l = np.zeros((NGrid))

h_v = (2*np.random.rand(N)-1)
psi_0 = np.zeros((2**N))
psi_0[0] = 1
#H = cm.h_tfim(h, J, N, h_v)
#H_T = -cm.h_tfim(h+dh, J, N, h_v)
H = cm.h_annni(h, J, J_prime, N, h_v)
H_T = -cm.h_annni(h+dh, J, J_prime, N,h_v)
#psi_0 = cm.psi_0_xxz(N)
#H = cm.h_xxz(h, J, N, d)
#H_T = -cm.h_xxz(h+dh, J, N, d)

psi_0_dagg = np.conjugate(np.transpose(psi_0))

e, v = la.eigh(H)
v_dagg = np.transpose(np.conjugate(v))

e_T, v_T = la.eigh(H_T)
v_dagg_T = np.transpose(np.conjugate(v_T))

psi_tau = cm.psi_at_t(e,v,v_dagg,tau,psi_0)

for i in range(NGrid):
		if dt[i]<=tau:
			avg_m_z[i], avg_s_c[i], b_fl[i], S[i], l[i] = time_evol(e,v,v_dagg,dt[i],psi_0,psi_0_dagg)
		else:
			avg_m_z[i], avg_s_c[i], b_fl[i], S[i], l[i] = time_evol(e_T,v_T,v_dagg_T,dt[i]-tau,psi_tau,psi_0_dagg)


'''
plt.subplot(4,1,1)
plt.plot(dt, avg_m_z)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$\langle m_z \rangle$", fontsize=18)
plt.subplot(4,1,2)
plt.plot(dt, S)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$S(t)$", fontsize=18)
plt.subplot(4,1,3)
plt.plot(dt, avg_s_c)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$\langle s_c \rangle $", fontsize=18)
plt.subplot(4,1,4)
plt.plot(dt, l)
plt.xticks([0,50,100,150,200,250], [0,50,r"$\tau$",150,r"$2\tau$",250])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$l(t)$", fontsize=18)
plt.show()
'''

plt.subplot(5,1,1)
plt.plot(dt, avg_m_z)
plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
plt.ylabel(r"$\langle m_z \rangle$", fontsize=18)
plt.subplot(5,1,2)
plt.plot(dt, S)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$S(t)$", fontsize=18)
plt.subplot(5,1,3)
plt.plot(dt, b_fl)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$\mathcal{F}$", fontsize=18)
plt.subplot(5,1,4)
plt.plot(dt, avg_s_c)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$\langle s_c \rangle$", fontsize=18)
plt.subplot(5,1,5)
plt.plot(dt, l)
plt.xticks([0,50,100,150,200,250], [0,50,r"$\tau$",150,r"$2\tau$",250])
#plt.yticks([0,-0.2,-0.4,-0.6,-0.8,-1])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$l(t)$", fontsize=18)
plt.subplots_adjust(hspace=.4)
plt.show()