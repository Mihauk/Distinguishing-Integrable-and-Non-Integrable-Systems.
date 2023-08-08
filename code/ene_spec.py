import numpy as np
from numpy import linalg as la
import common as cm
import pickle
import sys
from multiprocessing import Pool

def f(x):

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
	seed = 10
	assert N%2==0
	r = int(2**(N/2))
	np.random.seed(31415+seed)
	dtau = np.logspace(0,tau_max,tau_Grid)

	in_energy = np.zeros((samples))
	fi_energy = np.zeros((samples))

	H = cm.h_annni(h, J, J_prime, N)
	H_T = -cm.h_annni(h+dh, J, J_prime, N)
	
	e, v = la.eigh(H)
	v_dagg = np.transpose(np.conjugate(v))

	e_T, v_T = la.eigh(H_T)
	v_dagg_T = np.transpose(np.conjugate(v_T))

	dtf = np.linspace(0,2.5*dtau[tau_Grid-1], NGrid)

	for j in range(samples):
		print(j)
		#sys.stdout.flush()
		psi_0 = np.zeros((2**N))
		c = np.random.randint(2**N)
		psi_0[c] = 1
		psi_0_dagg = np.conjugate(psi_0.T)

		in_energy[j] = cm.check_real(np.dot(psi_0_dagg,np.dot(H,psi_0)))
		psi_fi = cm.psi_at_t(e_T,v_T,v_dagg_T,dtf[NGrid-1]-dtau[tau_Grid-1],cm.psi_at_t(e,v,v_dagg,dtau[tau_Grid-1],psi_0))
		psi_fi_dagg = np.conjugate(psi_fi.T)
		fi_energy[j] = cm.check_real(np.dot(psi_fi_dagg,np.dot(H_T,psi_fi)))
		print(j)

	with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/ene/annni_dh01_ran_inst_N'+str(N)+'_J'+str(J_prime)+'_seed'+str(seed)+'_tau_energy.pickle', 'wb') as data:
		pickle.dump([dtau, in_energy, fi_energy], data)


if __name__=='__main__':
	p = Pool(10)
	p.map(f,np.linspace(0,1,40))
