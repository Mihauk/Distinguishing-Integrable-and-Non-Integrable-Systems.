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
	#ddelta = 0.1
	tau_max=4
	#t_max = 2.5*(10**tau_max)
	samples = 100 #2**N
	seed = 80
	NGrid = 300
	tau_Grid = 40
	assert N%2==0
	r = int(2**(N/2))
	np.random.seed(31415+seed)
	dtau = np.logspace(2,tau_max,tau_Grid)

	m_z = cm.magnetization(N)
	s_c = cm.spin_corr(N)
	s_l2 = cm.sl2(N)

	dtf = np.linspace(0,2.5*dtau[tau_Grid-1], NGrid)
	window = (np.where(np.logical_and(dtf>=1.75*dtau[39],dtf<=2.25*dtau[39])))[0]
	
	avg_m_z = np.zeros((tau_Grid, samples, window.size))
	S = np.zeros((tau_Grid, samples, window.size))
	b_fl = np.zeros((tau_Grid, samples, window.size))
	avg_s_c = np.zeros((tau_Grid, samples, window.size))
	l = np.zeros((tau_Grid, samples, window.size))
	in_energy = np.zeros((samples))
	fi_energy = np.zeros((samples))

	for j in range(samples):
		print(j)
		#sys.stdout.flush()
		h_v = 0.1*(2*np.random.rand(N)-1)
		psi_0 = np.zeros((2**N))
		psi_0[0] = 1
		H = cm.h_annni(h, J, J_prime, N, h_v)
		H_T = -cm.h_annni(h+dh, J, J_prime, N, h_v)


		psi_0_dagg = np.conjugate(np.transpose(psi_0))

		e, v = la.eigh(H)
		v_dagg = np.transpose(np.conjugate(v))

		e_T, v_T = la.eigh(H_T)
		v_dagg_T = np.transpose(np.conjugate(v_T))

		in_energy[j] = cm.check_real(np.dot(psi_0_dagg,np.dot(H,psi_0)))
		psi_fi = cm.psi_at_t(e_T,v_T,v_dagg_T,dtf[NGrid-1]-dtau[tau_Grid-1],cm.psi_at_t(e,v,v_dagg,dtau[tau_Grid-1],psi_0))
		psi_fi_dagg = np.conjugate(psi_fi.T)
		fi_energy[j] = cm.check_real(np.dot(psi_fi_dagg,np.dot(H_T,psi_fi)))

		for k in range(tau_Grid):
			dt = np.linspace(0, 2.5*dtau[k], NGrid)
			psi_tau = cm.psi_at_t(e,v,v_dagg,dtau[k],psi_0)
			m=0
			for i in window:
				#if dt[i]<=dtau[k]:
				#	avg_m_z[k,j,i], avg_s_c[k,j,i], b_fl[k,j,i], S[k,j,i], l[k,j,i] = time_evol(e,v,v_dagg,dt[i],psi_0,psi_0_dagg)
				#else:
				avg_m_z[k,j,m], avg_s_c[k,j,m], b_fl[k,j,m], S[k,j,m], l[k,j,m] = time_evol(e_T,v_T,v_dagg_T,dt[i]-dtau[k],psi_tau,psi_0_dagg)
				m = m + 1

	with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/annni_hv01_N'+str(N)+'_J'+str(J_prime)+'_seed'+str(seed)+'_tau_energy_01.pickle', 'wb') as data:
		pickle.dump([dtau, avg_m_z, S, avg_s_c, l, b_fl, in_energy, fi_energy], data)

if __name__=='__main__':
	p = Pool(2)
	p.map(f, [0.1])#np.linspace(0,1,40))
