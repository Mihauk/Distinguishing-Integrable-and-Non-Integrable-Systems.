import numpy as np
from numpy import linalg as la
import common as cm
import pickle
import sys
from multiprocessing import Pool


def f(x):

	def time_evol(e, v, v_dagg, t, psi_0, psi_0_dagg):
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
	J_prime = x
	h = 1
	#d = 1
	dh = 0.1
	#dd = 0.1
	tau_max=2
	#t_max = 2.5*(10**tau_max)
	samples = 100
	NGrid = 300
	tau_Grid = 40
	assert N%2==0
	r = int(2**(N/2))

	dtau = np.logspace(0,tau_max,tau_Grid)

	m_z = cm.magnetization(N)
	s_c = cm.spin_corr(N)
	s_l2 = cm.sl2(N)

	avg_m_z = np.zeros((tau_Grid, samples, NGrid))
	S = np.zeros((tau_Grid, samples, NGrid))
	b_fl = np.zeros((tau_Grid, samples, NGrid))
	avg_s_c = np.zeros((tau_Grid, samples, NGrid))
	l = np.zeros((tau_Grid, samples, NGrid))

	H = cm.h_annni(h, J, J_prime, N)
	H_T = -cm.h_annni(h+dh, J, J_prime, N)
	#H = cm.h_tfim(h, J, N, h_v)
	#H_T = -cm.h_tfim(h+dh, J, N, h_v)
	#psi_0 = cm.psi_0_xxz(N)
	#H = cm.h_xxz(h, J, N, d, h_v)
	#H_T = -cm.h_xxz(h+dh, J, N, d, h_v)

	e, v = la.eigh(H)
	v_dagg = np.transpose(np.conjugate(v))

	e_T, v_T = la.eigh(H_T)
	v_dagg_T = np.transpose(np.conjugate(v_T))



	for j in range(samples):
		print(j)
		#sys.stdout.flush()
		psi_0 = np.zeros((2**N))
		c = np.random.randint(2**N)
		psi_0[c] = 1
		
		psi_0_dagg = np.conjugate(np.transpose(psi_0))

		
		for k in range(tau_Grid):
			dt = np.linspace(0, 2.5*dtau[k], NGrid)
			psi_tau = cm.psi_at_t(e,v,v_dagg,dtau[k],psi_0)
			for i in range(NGrid):
				if dt[i]<=dtau[k]:
					avg_m_z[k,j,i], avg_s_c[k,j,i], b_fl[k,j,i], S[k,j,i], l[k,j,i] = time_evol(e,v,v_dagg,dt[i],psi_0,psi_0_dagg)
				else:
					avg_m_z[k,j,i], avg_s_c[k,j,i], b_fl[k,j,i], S[k,j,i], l[k,j,i] = time_evol(e_T,v_T,v_dagg_T,dt[i]-dtau[k],psi_tau,psi_0_dagg)

	with open('../data/new/annni_N'+str(N)+'_J'+str(J_prime)+'_tau.pickle', 'wb') as data:
		pickle.dump([dt, dtau, avg_m_z, S, avg_s_c, l, b_fl], data)

if __name__=='__main__':
	p = Pool(10)
	p.map(f,np.linspace(0,1,40))