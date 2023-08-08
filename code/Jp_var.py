import numpy as np
from numpy import linalg as la
import common as cm
import pickle

N = 10
J = 1
J_prime_max = 1
h = 1
d = 1
dh = 0.1
dd = 0.1
tau=100
t_max = 2.5*tau
samples = 100
NGrid = 300
Jp_Grid = 40
assert N%2==0
r = int(2**(N/2))

dt = np.linspace(0., t_max, NGrid)
dJp = np.linspace(0,J_prime_max,Jp_Grid)

m_z = cm.magnetization(N)
s_c = cm.spin_corr(N)
avg_m_z = np.zeros((Jp_Grid, samples, NGrid))
S = np.zeros((Jp_Grid, samples, NGrid))
b_fl = np.zeros((tau_Grid, samples, NGrid))
avg_s_c = np.zeros((Jp_Grid, samples, NGrid))
l = np.zeros((Jp_Grid, samples, NGrid))

for k in range(Jp_Grid):
	for j in range(samples):
		print(j)
		h_v = 0.1*(2*np.random.rand(N)-1)
		psi_0 = np.zeros((2**N))
		psi_0[0] = 1
		#H = cm.h_tfim(h, J, N, h_v)
		#H_T = -cm.h_tfim(h+dh, J, N, h_v)
		H = cm.h_annni(h, J, dJp[k], N, h_v)
		H_T = -cm.h_annni(h+dh, J, dJp[k], N, h_v)
		#psi_0 = cm.psi_0_xxz(N)
		#H = cm.h_xxz(h, J, N, d, h_v)
		#H_T = -cm.h_xxz(h+dh, J, N, d, h_v)

		psi_0_dagg = np.conjugate(np.transpose(psi_0))

		e, v = la.eigh(H)
		v_dagg = np.transpose(np.conjugate(v))

		e_T, v_T = la.eigh(H_T)
		v_dagg_T = np.transpose(np.conjugate(v_T))

		for i in range(NGrid):
			if dt[i]<=tau:
				U = cm.exp_m(e,v,v_dagg,dt[i])
				psi_t = np.dot(U,psi_0)
				psi_t_dagg = np.transpose(np.conjugate(psi_t))
				avg_m_z[k][j][i] = cm.check_real(np.dot(psi_t_dagg,np.dot(m_z,psi_t)))
				avg_s_c[k][j][i] = cm.check_real(np.dot(psi_t_dagg,np.dot(s_c,psi_t)))
				b_fl[k][j][i] = cm.bipartite_fl(psi_t, psi_t_dagg, N)
				S[k][j][i] = cm.entropy(psi_t,r)
				l[k][j][i] = cm.fidelity(psi_0_dagg,psi_t,N)
			else:
				U = np.dot(cm.exp_m(e_T,v_T,v_dagg_T,(dt[i]-tau)), cm.exp_m(e,v,v_dagg,tau))
				psi_t = np.dot(U,psi_0)
				psi_t_dagg = np.transpose(np.conjugate(psi_t))
				avg_m_z[k][j][i] = cm.check_real(np.dot(psi_t_dagg,np.dot(m_z,psi_t)))
				avg_s_c[k][j][i] = cm.check_real(np.dot(psi_t_dagg,np.dot(s_c,psi_t)))
				b_fl[k][j][i] = cm.bipartite_fl(psi_t, psi_t_dagg, N)
				S[k][j][i] = cm.entropy(psi_t,r)
				l[k][j][i] = cm.fidelity(psi_0_dagg,psi_t,N)

with open('../data/annni_hv01_N10_J01.pickle', 'wb') as data:
	pickle.dump([dt, dJp, avg_m_z, S, avg_s_c, l, b_fl], data)