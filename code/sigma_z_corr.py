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
		#P = P**l
		#for ii in range(l):
		P = np.dot(P,(I - 2j*cm.spin("Y", (1+c+m), N))) #
	return (P)

N = 8
c = 3
d = 4
#l = 1
J = 1
#J_p = 0.1*J
t_max = 5
Ngrid = 200
samples = 50
dis_sam = 100
#dt = np.linspace(0, t_max, Ngrid)
dt = np.logspace(-1, t_max, Ngrid)

I = cm.spin("I", 0, N)

sigma_z_1 = cm.spin("+", c, N)
D = np.zeros((dis_sam, samples, d, Ngrid))

for m in range(d):
	print(m)
	#sys.stdout.flush()
	sigma_z_2 = cm.spin("-", c+m+1, N)
	P_pio2_1 = pulse(1,m)
	P_pio2_2 = pulse(2,m)
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
			r = np.random.randint(v[0].size)
			psi_0 = v[:,r]
			#print(psi_0)
			phi_0 = np.dot(P_pio2_1,psi_0)
			for i in range(Ngrid):
				#phi_t = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
				phi_t = np.dot(P_pi_1,np.dot(P_pi_2,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
				chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
				chi_t_dagg = np.conj(chi_t.T)
				D[k, j, m, i] = (np.dot(chi_t_dagg, np.dot(sigma_z_1, np.dot(sigma_z_2,chi_t))) - (np.dot(chi_t_dagg, np.dot(sigma_z_1, chi_t))*np.dot(chi_t_dagg, np.dot(sigma_z_2, chi_t))))

with open('/home/abhishek/Documents/ny project/data/sig_plus_min_corr/hv10_N'+str(N)+'_eav50_ovr_egns_dis100_t100000_d_rel_tsi_s14_Ngrid200_siz_plus_mins_corr.pickle', 'wb') as data:
	pickle.dump([dt, D], data)