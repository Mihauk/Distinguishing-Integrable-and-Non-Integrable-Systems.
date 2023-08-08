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

def psi_prd_st():
	q = 1
	r = np.random.randint(2, size=N)
	for ii in range(N):
		if r[ii] == 1:
			q = np.kron(q,cm.up)
		else:
			q = np.kron(q,cm.down)
	return q	

N = 8
c = 1
d = 6
J = 1
t_max = 8
Ngrid = 200
#seed = 50
samples = 256
dis_sam = 100
#dt = np.linspace(0, t_max, Ngrid)
dt = np.logspace(-1, t_max, Ngrid)

I = cm.spin("I", 0, N)
#np.random.seed(31415+seed)
sigma_z_1 = cm.spin("Z", c, N)
S = np.zeros((dis_sam, samples, Ngrid))
#D = np.zeros((dis_sam, samples, Ngrid))
#m=5

D = np.zeros((dis_sam, samples, d-1, Ngrid))

for m in range(1,d):
	print(m)
	#sys.stdout.flush()
	P_pio2_1 = pulse(1,m)
	P_pio2_2 = pulse(2,m)
	P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
	P_pi_2 = np.dot(P_pio2_2, P_pio2_2)
	for j in range(samples):
		print(j)
		s = 1
		#r = np.random.randint(v[0].size)
		r = np.zeros((2**N))
		r[j] = 1
		psi_0 = r #psi_prd_st()	#v[:,r]
		psi_0_dagg = np.conj(psi_0.T)
		mz = np.dot(psi_0_dagg, np.dot(2*sigma_z_1, psi_0))
		if (mz<0):
			s = -1
		phi_0 = np.dot(P_pio2_1,psi_0)
		for k in range(dis_sam):
			#print(k)
			h_v = 7.5*(2*np.random.rand(N)-1)
			H = cm.h_rfh_obc( J, N, h_v)
			e, v = la.eigh(H)
			v_dagg = np.conj(v.T)
			for i in range(Ngrid):
				if (m==1):
					phi_t_s = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
					chi_t_s = np.dot(P_pio2_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t_s))
					chi_t_s_dagg = np.conj(chi_t_s.T)
					S[k,j,i] = cm.check_real(s*np.dot(chi_t_s_dagg, np.dot(sigma_z_1,chi_t_s)))
				phi_t = np.dot(P_pi_1,np.dot(P_pi_2, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
				chi_t = np.dot(P_pio2_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
				chi_t_dagg = np.conj(chi_t.T)
				D[k, j, m-1, i] = cm.check_real(s*np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhiraj654/Documents/data_paper_deer/spin/hv7.5_N'+str(N)+'_eav50_dis100_t108_tsi_s12_d1-5_spin_echo_Ngrid200.pickle', 'wb') as data:
	pickle.dump([dt, D, S], data)