import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

def pulse(i, m):
	P = 1/np.sqrt(2)
	if i == 1:
		P = np.dot(P,(I - 2j*cm.spin("Y", c, N)))
	elif i == 2:
		P = np.dot(P,(I - 2j*cm.spin("Y", (1+c+m), N)))
	return (P)

def Ham(h_v):
	hamiltonian = np.zeros((hilbert,hilbert))
	for i in range(N-1):
		hamiltonian += (np.kron(np.eye(2**i), np.kron(two_body, np.eye(2**(N-i-2)))))
		hamiltonian += h_v[i]*(np.kron(np.eye(2**i), np.kron(sz, np.eye(2**(N-i-1)))))
	hamiltonian += h_v[N-1]*(np.kron(np.eye(2**(N-1)), sz))
	return hamiltonian

sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0,-1.0j],[1.0j,0.0]])
sz = np.array([[1.0, 0.0], [0.0,-1.0]])

N = 8
c = 1
d = 5
J = 1
t_max = 8
Ngrid = 200
#seed = 50
samples = 50
dis_sam = 500
hilbert = 2**N

dt = np.logspace(-1, t_max, Ngrid)

I = cm.spin("I", 0, N)
#np.random.seed(31415+seed)
sigma_z_1 = np.kron(np.eye(2**c), np.kron(sz, np.eye(2**(N-c-1))))
S = np.zeros((dis_sam, samples, Ngrid))
D = np.zeros((dis_sam, samples, d, Ngrid))

two_body = J*(np.kron(sx,sx) + np.kron(sy,sy) + np.kron(sz,sz))
assert np.allclose(two_body.imag, 0)
two_body = two_body.real

for m in range(d):
	print(m)
	#sys.stdout.flush()
	P_pio2_1 = pulse(1,m)
	P_pio2_2 = pulse(2,m)
	P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
	P_pi_2 = np.dot(P_pio2_2, P_pio2_2)
	for k in range(dis_sam):
		print(k)
		h_v = 7.5*(2*np.random.rand(N)-1)
		H = Ham(h_v)
		e, v = la.eigh(H)
		v_dagg = np.conj(v.T)

		for j in range(samples):
			s = 1
			r = np.random.randint(v[0].size)
			psi_0 = v[:,r]
			psi_0_dagg = np.conj(psi_0.T)
			mz = np.dot(psi_0_dagg, np.dot(sigma_z_1, psi_0))
			if (mz<0):
				s = -1
			phi_0 = np.dot(P_pio2_1,psi_0)
			for i in range(Ngrid):
				if (m==0):
					phi_t_s = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
					chi_t_s = np.dot(P_pio2_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t_s))
					chi_t_s_dagg = np.conj(chi_t_s.T)
					S[k,j,i] = cm.check_real(s*np.dot(chi_t_s_dagg, np.dot(sigma_z_1,chi_t_s)))
				phi_t = np.dot(P_pi_1,np.dot(P_pi_2, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
				chi_t = np.dot(P_pio2_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
				chi_t_dagg = np.conj(chi_t.T)
				D[k, j, m, i] = cm.check_real(s*np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhiraj654/Documents/data_paper_deer/pauli/hv7.5_N'+str(N)+'_eav50_ovr_egns_dis500_t108_d_rel_tsi_s12_d5_spin_echo_Ngrid200.pickle', 'wb') as data:
	pickle.dump([dt, D, S], data)