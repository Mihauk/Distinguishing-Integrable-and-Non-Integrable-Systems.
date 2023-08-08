import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

def pulse(m):
	P = 1/np.sqrt(2)
	P = np.dot(P,(I - 2j*cm.spin("Y", m, N)))
	return (P)

N = 8
c = 4
J = 1
#J_p = 0.1*J
t_max = 5
Ngrid = 200
samples = 50
dis_sam = 100
#dt = np.linspace(0, t_max, Ngrid)
dt = np.logspace(-1, t_max, Ngrid)

I = cm.spin("I", 0, N)


D = np.zeros((dis_sam, samples, c, Ngrid))

for m in range(c):
	print(m)
	sigma_z_1 = cm.spin("Z", m, N)
	P_pio2_1 = pulse(m)
	P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
	for k in range(dis_sam):
		print(k)
		h_v = (10*np.random.rand(N)-5)
		H = cm.h_rfh_obc( J, N, h_v)
		#H = cm.h_rfxxz_obc( J, J_p, N, h_v)
		e, v = la.eigh(H)
		v_dagg = np.conj(v.T)

		for j in range(samples):
			s = 1
			r = np.random.randint(2, size=N)
			r[m] = 1
			for ii in range(N):
				if r[ii] == 1:
					s = np.kron(s,cm.up)
				else:
					s = np.kron(s,cm.down)
			psi_0 = s
			phi_0 = np.dot(P_pio2_1,psi_0)
			for i in range(Ngrid):
				phi_t = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
				chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
				chi_t_dagg = np.conj(chi_t.T)
				D[k, j, m, i] = cm.check_real(np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhishek/Documents/ny project/data/hv5_N'+str(N)+'_eav50_dis100_t100000_s1to4_rc_Ngrid200_spin_echo.pickle', 'wb') as data:
	pickle.dump([dt, D], data)