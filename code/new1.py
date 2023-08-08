import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

def pulse(i):
	P = 1/np.sqrt(2)
	if i == 1:
		P = np.dot(P,(I - 2j*cm.spin("Y", c, N)))
	elif i == 2:
		#P = P**l
		#for ii in range(l):
		P = np.dot(P,(I - 2j*cm.spin("Y", (1+c+d), N))) #
	return (P)

N = 8
c = 0
d = 6
J = 1
#J_p = 0.1*J
t_max = 12
Ngrid = 200
samples = 50
dis_sam = 1000
#dt = np.linspace(0, t_max, Ngrid)
dt = np.logspace(-1, t_max, Ngrid)

I = cm.spin("I", 0, N)

sigma_z_1 = cm.spin("Z", c, N)
sp = np.zeros((dis_sam, samples, Ngrid))
D = np.zeros((dis_sam, samples, Ngrid))

sigma_z_2 = cm.spin("Z", c+d+1, N)
P_pio2_1 = pulse(1)
P_pio2_2 = pulse(2)
P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
P_pi_2 = np.dot(P_pio2_2, P_pio2_2)
for k in range(dis_sam):
	print(k)
	h_v = 10*(2*np.random.rand(N)-1)
	H = cm.h_rfh_obc( J, N, h_v)
	#H = cm.h_rfxxz_obc( J, J_p, N, h_v)
	e, v = la.eigh(H)
	v_dagg = np.conj(v.T)

	for j in range(samples):
		s = 1
		r = np.random.randint(2, size=N)
		r[c] = 1
		for ii in range(N):
			if r[ii] == 1:
				s = np.kron(s,cm.up)
			else:
				s = np.kron(s,cm.down)
		psi_0 = s
		phi_0 = np.dot(P_pio2_1,psi_0)
		for i in range(Ngrid):
			phi_t_s = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
			chi_t_s = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t_s))
			chi_t_s_dagg = np.conj(chi_t_s.T)
			sp[k, j, i] = cm.check_real(np.dot(chi_t_s_dagg, np.dot(sigma_z_1,chi_t_s)))
			phi_t = np.dot(P_pi_1,np.dot(P_pi_2,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
			chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
			chi_t_dagg = np.conj(chi_t.T)
			D[k, j, i] = cm.check_real(np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhishek/Documents/ny project/data/hv20_N'+str(N)+'_eav50_dis1000_t112_d_rel_tsi_spin_echo_s11_s28_Ngrid200_spa_corr.pickle', 'wb') as data:
	pickle.dump([dt, D, sp], data)