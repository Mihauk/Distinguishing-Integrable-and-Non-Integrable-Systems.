import numpy as np
from numpy import linalg as la
#import matplotlib.pyplot as plt
import common as cm
import pickle

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
h = 1
#d = 1
dh = 0.1
#dd = 0.1
tau=100
t_max = 2.5*tau
samples = 100
NGrid = 300
assert N%2==0
r = int(2**(N/2))

dt = np.linspace(0., t_max, NGrid)

m_z = cm.magnetization(N)
s_c = cm.spin_corr(N)
s_l2 = cm.sl2(N)

avg_m_z = np.zeros((samples,NGrid))
S = np.zeros((samples,NGrid))
b_fl = np.zeros((samples,NGrid))
avg_s_c = np.zeros((samples,NGrid))
l = np.zeros((samples,NGrid))

for j in range(samples):
	print(j)
	h_v = (2*np.random.rand(N)-1)
	psi_0 = np.zeros((2**N,1))
	psi_0[0] = 1
	H = cm.h_tfim(h, J, N, h_v)
	H_T = -cm.h_tfim(h+dh, J, N, h_v)
	#H = cm.h_annni(h, J, J_prime, N, h_v)
	#H_T = -cm.h_annni(h+dh, J, J_prime, N, h_v)
	#psi_0 = cm.psi_0_xxz(N)
	#H = cm.h_xxz(h, J, N, d, h_v)
	#H_T = -cm.h_xxz(h+dh, J, N, d, h_v)

	psi_0_dagg = np.conjugate(np.transpose(psi_0))

	e, v = la.eigh(H)
	v_dagg = np.transpose(np.conjugate(v))

	e_T, v_T = la.eigh(H_T)
	v_dagg_T = np.transpose(np.conjugate(v_T))

	psi_tau = cm.psi_at_t(e,v,v_dagg,tau,psi_0)

	for i in range(NGrid):
		if dt[i]<=tau:
			avg_m_z[j,i], avg_s_c[j,i], b_fl[j,i], S[j,i], l[j,i] = time_evol(e,v,v_dagg,dt[i],psi_0,psi_0_dagg)
		else:
			avg_m_z[j,i], avg_s_c[j,i], b_fl[j,i], S[j,i], l[j,i] = time_evol(e_T,v_T,v_dagg_T,dt[i]-tau,psi_tau,psi_0_dagg)

with open('../data/tfim_hv1_N'+str(N)+'_J'+str(J_prime)+'1.pickle', 'wb') as data:
	pickle.dump([dt, avg_m_z, S, avg_s_c, l, b_fl], data)