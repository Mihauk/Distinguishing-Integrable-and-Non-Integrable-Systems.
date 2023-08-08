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
c = 2
d = 3
l = 1
J = 1
#J_p = 0.1*J
t_max =  150 
Ngrid = 200
samples = 50
dis_sam = 50
dt = np.linspace(0, t_max, Ngrid)

I = cm.spin("I", 0, N)

P_pio2_1 = pulse(1)
P_pio2_2 = pulse(2)
P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
P_pi_2 = np.dot(P_pio2_2, P_pio2_2)

sigma_z_1 = cm.spin("Z", c, N)
D = np.zeros((dis_sam, samples, Ngrid))

for k in range(dis_sam):
	print(k)
	sys.stdout.flush()
	h_v = (10*np.random.rand(N)-5)
	H = cm.h_rfh_obc( J, N, h_v)
	#H = cm.h_rfxxz_obc( J, J_p, N, h_v)
	e, v = la.eigh(H)
	v_dagg = np.conj(v.T)

	for j in range(samples):
		s = 1
		r = np.random.randint(2, size=N)
		#r[c] = 1
		for ii in range(N):
			if r[ii] == 1:
				s = np.kron(s,cm.up)
			else:
				s = np.kron(s,cm.down)
		psi_0 = s
		phi_0 = np.dot(P_pio2_1,psi_0)
		for i in range(Ngrid):
			#phi_t = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
			phi_t = np.dot(P_pi_1,np.dot(P_pi_2,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
			chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
			chi_t_dagg = np.conj(chi_t.T)
			D[k, j, i] = cm.check_real(np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhishek/Documents/ny project/data/deer_hv5_N'+str(N)+'_tav_tsi.pickle', 'wb') as data: #two spin interaction
	pickle.dump([dt, D], data)